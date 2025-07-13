from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import logging
import re
from dotenv import load_dotenv
from supabase import create_client
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from upload_pdf import process_pdf
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from config import PINECONE_API_KEY
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from fetch_text_supabase import fetch_full_text_from_supabase
from langchain.schema import HumanMessage




# === Load Environment Variables ===
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bhrwvazkvsebdxstdcow.supabase.co/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

# === Initialize Logging ===
logging.basicConfig(level=logging.DEBUG)

# === Initialize Flask App ===
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app, resources={
    r"/chat": {"origins": "http://localhost:5173", "methods": ["POST"]},
    r"/upload": {"origins": "http://localhost:5173", "methods": ["POST"]},
    r"/ask": {"origins": "http://localhost:5173", "methods": ["POST"]},
})

# === Initialize Supabase & Langchain Models ===
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7)
memory = ConversationBufferWindowMemory(k=10, return_messages=True)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
embedding_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

# === Global Variable ===
recent_file_uid = None

# === Helper Functions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def insert_chat_log(user_message, response_message, user_id=None):
    try:
        data = {
            "user_message": user_message,
            "response_message": response_message,
            "user_id": user_id
        }
        response = supabase.table("chat_logs").insert(data).execute()
        logging.info("Inserted chat log into Supabase.")
    except Exception as e:
        logging.error(f"Supabase insert error: {e}")

# === Endpoints ===
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message", "")
        user_id = request.json.get("user_id")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        logging.info(f"Received message: {user_message}")

        # Enhanced educational and conversational prompt
        prompt = f"""
        You are an AI assistant with a deep knowledge base designed to help users learn and understand any topic in the most effective and engaging way. Your role is to provide clear, accurate, and detailed explanations, making complex topics easy to understand. 

        Respond to the user with a friendly, conversational tone, as if you're explaining the concept to a student. Break down the topic step by step when necessary, and give real-life examples to aid comprehension. Also, offer YouTube video suggestions that are relevant to the topic for further learning.

        Be empathetic, patient, and provide well-rounded answers. If the user asks for clarifications or examples, be ready to offer more detailed responses and give helpful suggestions.

        User message: {user_message}
        """

        # Run the conversation chain with memory
        ai_reply = conversation.predict(input=user_message)

        # Save to Supabase AFTER memory processes it
        insert_chat_log(user_message, ai_reply, user_id)

        return jsonify({"response": ai_reply}), 200

    except Exception as e:
        logging.error(f"/chat error: {e}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    global recent_file_uid 
    user_id=request.form.get("user_id")  # Get user ID from the form data
    print(f"User ID from upload: {user_id}")  # Debugging line to check user ID

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check file size
    if file and file.content_length > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": "File is too large. Max size is 5MB"}), 413

    if file and allowed_file(file.filename):
        # Save the file to the server
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        logging.info(f"📥 File saved: {file_path}")

        try:
            # Call your separate module to handle chunking + saving
            success, chunk_count, uploaded_filename, file_uid = process_pdf(file_path, supabase, GEMINI_API_KEY, user_id)

            # Save the file UUID to the global variable
            recent_file_uid = file_uid  # Store in global variable
            logging.info(f"File UUID stored: {recent_file_uid}")

            if success:
                logging.info(f"📌 recent_filename set to: {uploaded_filename}")
                logging.info(f"🗂️ File UUID: {file_uid}")
                return jsonify({"message": f"PDF processed. {chunk_count} chunks saved from '{uploaded_filename}'."}), 200
            else:
                return jsonify({"error": f"Failed to process PDF: {chunk_count}"}), 500

        except Exception as e:
            logging.error(f"Error during PDF processing: {str(e)}")
            return jsonify({"error": "Failed to process the PDF file."}), 500

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_message = request.json.get("message", "")  # Get the user's message
        user_id = request.json.get("user_id")
        
        # List of phrases that indicate the user wants a summary or overview
        summary_keywords = [
            "summarize", "overview", "summary", "what is on this file", "what is this file about", 
            "can you summarize", "give me a summary", "summarize the document", "overview of this file",
            "tell me about this file", "what's in this document", "can you give me the content of this file",
            "give me an overview", "give me the summary","summary of this file", "summarize this file", "summarize this document",
            "give me a summary of this file", "give me a summary of this document","summarise"
        ]
        
        # Expanded list of phrases indicating the user wants to generate questions
        question_keywords = [
            "generate questions","generate question", "make questions","make question", "create questions","create question", "ask questions", 
            "can you make questions for this document", "can you generate questions from this", 
            "create quiz questions", "generate quiz questions", "formulate questions from this", 
            "please make some questions", "please generate questions", "can you suggest questions", 
            "ask about this document", "can you create some questions", "create some questions for me", 
            "formulate questions", "suggest questions based on this", "can you create a quiz", 
            "can you ask some questions", "make some questions about this","create some questions","give me some questions",
            "generate questions from this file", "generate questions from this document", "make questions from this file",
            "make questions from this document", "create questions from this file", "create questions from this document",
            "ask questions about this file", "ask questions about this document", "can you generate questions from this file",
            "can you generate questions from this document", "can you make questions from this file", "can you make questions from this document",
        ]

        # If the user asks for a summary, trigger the summary function
        # if any(re.search(r'\b' + re.escape(keyword) + r'\b', user_message, re.IGNORECASE) for keyword in summary_keywords):
        #     return get_summary()

        # # If the user asks to generate questions, trigger question generation
        # elif any(re.search(r'\b' + re.escape(keyword) + r'\b', user_message, re.IGNORECASE) for keyword in question_keywords):
        #     return generate_questions()

        # For any other question, perform similarity search and provide an answer
        return get_answer_from_file(user_message, user_id)

    except Exception as e:
        logging.error(f"❌ Ask route error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
# === Functional Chains ===
def get_summary():
    try:
        global recent_file_uid

        if not recent_file_uid:
            return jsonify({"error": "No recent file uploaded"}), 400

        response = supabase.table('documents').select('id', 'content').eq('file_uuid', recent_file_uid).execute()
        docs = response.data

        if not docs:
            return jsonify({"error": "No content found for the given file."}), 404

        concatenated_text = "\n\n".join(doc['content'] for doc in docs)

        prompt = f"""
        You are an expert assistant with a deep understanding of how to summarize complex documents in a clear, concise, and professional manner. Based on the following content, summarize the key points in a way that is easy to understand, highlighting the most important information while keeping the summary brief and to the point.

        Please focus on:
        1. Extracting the core ideas and themes.
        2. Presenting the summary in a structured format with key takeaways.
        3. Avoiding unnecessary details or long explanations.

        Make sure the summary is **short and professional**, providing only the most relevant and actionable insights.

        Here is the content you need to summarize:

        ======== PDF Content ========
        {concatenated_text}
        =============================

        Your summary:
        """

        summary = llm.predict(prompt)
        summary = summary.strip()

        # Save summary to the first document row associated with this file
        first_id = docs[0]['id']
        supabase.table("documents").update({"summary": summary}).eq("id", first_id).execute()

        # Optional: log the interaction
        insert_chat_log("Summarize Request", summary)

        return jsonify({"response": summary}), 200

    except Exception as e:
        logging.error(f"Error in get_summary: {e}")
        return jsonify({"error": "Failed to summarize"}), 500

def generate_questions():
    global recent_file_uid
    try:
        if not recent_file_uid:
            return jsonify({"error": "No recent file uploaded"}), 400

        docs = supabase.table('documents').select('content').eq('file_uuid', recent_file_uid).execute().data
        if not docs:
            return jsonify({"error": "No content found"}), 404

        concatenated_text = "\n\n".join(doc['content'] for doc in docs)
        
        prompt = f"""
        You are an expert at generating questions from a given text. Based on the following document, create relevant and insightful questions that can be asked:You are an expert educator with a deep understanding of how to generate relevant and insightful questions from a given text. Based on the following document, create a mixture of **Multiple Choice Questions (MCQs)** and **broad, open-ended questions** that focus on the most important topics discussed in the text. These questions should reflect the depth and complexity of the material, similar to what a professional-grade teacher would ask.

        For each question:

        1. Provide the **correct answer** at the end of the question on a **separate line**.
        2. **Explain why** the answer is correct for **broad questions**, with a focus on the key concepts behind the question. Keep the explanation concise, just a few sentences.
        3. Specify **which topic** the question is related to on a **separate line with a line gap** (e.g., "Topic: [Topic Name]").
        4. Ensure the questions are well-structured, engaging, and relevant to the content.
        5. If applicable, provide at least **one resource or website** where users can learn more about each topic (this is **optional** and should be on a **separate new line with a line gap** if included).
        6. **For MCQs**, do not provide explanations—just the question and the answer on separate lines.
        7. **For broad questions**, make sure the explanation is concise, focusing on key ideas.
        8. Options will be on a separate line and make gap between the 4 options as well as the question so that it is easy to read.
        9. It is not mandatory to generate 10 questions, but try to generate at least 5 questions.

        Here is the content you need to generate questions from:

        ======== PDF Content ========
        {concatenated_text}
        =============================

        Generated Questions and Answers will be in the following format:

        1. **Question**: [Insert Question Here]

        **Answer**: [Insert Correct Answer Here]

        **Topic**: [Insert Topic Name Here]

        [Optional: **Learn More**: [Insert Learning Resource or Website Here]]

        2. **Question**: [Insert Question Here]

        **Answer**: [Insert Correct Answer Here]

        **Explanation**: [Provide Explanation for the Answer in a Few Sentences]

        **Topic**: [Insert Topic Name Here]

        [Optional: **Learn More**: [Insert Learning Resource or Website Here]]

        [Repeat as needed for additional questions]
        """
        questions = llm.predict(prompt)
        insert_chat_log("Generate Questions Request", questions.strip())
        return jsonify({"response": questions.strip()}), 200
    except Exception as e:
        logging.error(f"Error in generate_questions: {e}")
        return jsonify({"error": "Failed to generate questions"}), 500

def get_answer_from_file(user_query, user_id):
    try:
        # # Get input from the request body
        # user_query = request.json.get("user_query")
        # user_id = request.json.get("user_id")
        userID=user_id;
        userQuery= user_query;

        if not user_query or not user_id:
            return jsonify({"error": "Missing user_query or user_id"}), 400


        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Get existing index
        index = pc.Index("document-index")

        # Set up Langchain Pinecone VectorStore
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_fn
        )

        query_embedding = embedding_fn.embed_query(userQuery)

        pinecone_results = index.query(
            vector=query_embedding,
            filter={"user_id": userID},
            top_k=5,
            include_metadata=True
        )

        print(f"Pinecone Matches: {pinecone_results['matches']}")
        # Check if no matches found
        if not pinecone_results['matches']:
            return jsonify({"response": "No relevant content found."}), 200


        relevant_docs = []
        for match in pinecone_results['matches']:
            print(f"Metadata: {match['metadata']}")
            file_uuid = match['metadata'].get("file_uuid")
            full_text = fetch_full_text_from_supabase(supabase, file_uuid, user_id)
            print(f"Full Text: {full_text}")
            
        if full_text:
            relevant_docs.append(full_text)

        if relevant_docs:
            # Concatenate or process relevant_docs as needed before LLM
            combined_text = "\n".join(relevant_docs)
            # Build chat message properly
            prompt = combined_text + "\n\nUser Question: " + user_query
            result = llm([HumanMessage(content=prompt)])

            return jsonify({
                "response": result.content,
                "source_documents": relevant_docs
            }), 200
        else:
            return jsonify({"response": "No relevant content found."}), 200

    except Exception as e:
        logging.error(f"Error in get_answer_from_file: {e}")
        return jsonify({"error": "Failed to fetch answer"}), 500

def get_explanation_from_api(user_query):
    try:
        reply = llm.predict(user_query)
        return jsonify({"response": reply.strip()}), 200
    except Exception as e:
        logging.error(f"Error in get_explanation_from_api: {e}")
        return jsonify({"error": "Fallback failed"}), 500

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)