from flask import Flask, request, jsonify,current_app
from flask_cors import CORS
from flask.testing import FlaskClient
import os
import logging
from supabase import create_client
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from upload_pdf import process_pdf  
import re


# Global variable to store the recent file UUID
recent_file_uid = None


# === Load .env and setup ===
load_dotenv()

SUPABASE_URL = "https://bhrwvazkvsebdxstdcow.supabase.co/"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

logging.basicConfig(level=logging.DEBUG)

# === Initialize Flask App ===
app = Flask(__name__)
CORS(app, resources={
    r"/chat": {"origins": "http://localhost:5173", "methods": ["POST"]},
    r"/upload": {"origins": "http://localhost:5173", "methods": ["POST"]},
    r"/ask": {"origins": "http://localhost:5173", "methods": ["POST"]},  # ✅ ADD THIS LINE
})
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Initialize Supabase ===
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Initialize LangChain Gemini Chat Model ===
# chat_model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",  # or "gemini-1.5-pro"
#     google_api_key=GEMINI_API_KEY,
#     temperature=0.7
# )

# Setup Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

# Embedding model
embedding_fn = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)


# === File type check ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Supabase logging ===
def insert_chat_log(user_message, response_message):
    try:
        data = {"user_message": user_message, "response_message": response_message}
        response = supabase.table("chat_logs").insert(data).execute()
        # Check if there is an error in the response
        if response.data is None:
            logging.warning(f"Supabase insert failed: {response.error.message}")
        else:
            logging.info("Message successfully inserted into Supabase.")
    except Exception as e:
        logging.error(f"Supabase insert error: {str(e)}")
        
# === Gemini Chat Endpoint ===
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message", "")
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

        # Send the prompt to Gemini
        reply = llm.predict(prompt)

        if not reply:
            return jsonify({"error": "No response from Gemini"}), 500

        logging.info(f"Generated reply: {reply}")
        insert_chat_log(user_message, reply)

        return jsonify({"response": reply}), 200

    except Exception as e:
        logging.error(f"/chat error: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500







@app.route('/upload', methods=['POST'])
def upload_file():
    global recent_file_uid  # Ensure to use the global variable

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

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
            success, chunk_count, uploaded_filename, file_uid = process_pdf(file_path, supabase, GEMINI_API_KEY)

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
            "generate questions", "make questions", "create questions", "ask questions", 
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
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', user_message, re.IGNORECASE) for keyword in summary_keywords):
            return get_summary()

        # If the user asks to generate questions, trigger question generation
        elif any(re.search(r'\b' + re.escape(keyword) + r'\b', user_message, re.IGNORECASE) for keyword in question_keywords):
            return generate_questions()

        # For any other question, perform similarity search and provide an answer
        return get_answer_from_file(user_message)

    except Exception as e:
        logging.error(f"❌ Ask route error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

    
def get_summary():
    try:
        global recent_file_uid

        if not recent_file_uid:
            return jsonify({"error": "No recent file uploaded"}), 400

        # Directly query Supabase to retrieve the chunks for the given file_uuid
        response = supabase.table('documents').select('content').eq('file_uuid', recent_file_uid).execute()
        docs = response.data

        if not docs:
            return jsonify({"error": "No content found for the given file."}), 404

        # Concatenate the content of all chunks into one string
        concatenated_text = "\n\n".join(doc['content'] for doc in docs)

        # Pass the concatenated content to the summarization model
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

        try:
            summary = llm.predict(prompt)  # Use the summarization model (e.g., Gemini)
            insert_chat_log("Summarize Request", summary.strip())  # Log the interaction
            return jsonify({"response": summary.strip()}), 200  # Return the summary
        except Exception as e:
            logging.error(f"❌ Error summarizing the content: {str(e)}")
            return jsonify({"response": "⚠️ Failed to summarize the content."}), 200
    except Exception as e:
        logging.error(f"❌ Error in get_summary: {str(e)}")
        return jsonify({"error": "Internal error occurred while fetching the summary."}), 500


def generate_questions():
    try:
        global recent_file_uid

        if not recent_file_uid:
            return jsonify({"error": "No recent file uploaded"}), 400

        # Directly query Supabase to retrieve the chunks for the given file_uuid
        response = supabase.table('documents').select('content').eq('file_uuid', recent_file_uid).execute()
        docs = response.data

        if not docs:
            return jsonify({"error": "No content found for the given file."}), 404

        # Concatenate the content of all chunks into one string
        concatenated_text = "\n\n".join(doc['content'] for doc in docs)

        # Pass the concatenated content to a model to generate questions
        prompt = f"""
        You are an expert at generating questions from a given text. Based on the following document, create relevant and insightful questions that can be asked:You are an expert educator with a deep understanding of how to generate relevant and insightful questions from a given text. Based on the following document, create a mixture of **Multiple Choice Questions (MCQs)** and **broad, open-ended questions** that focus on the most important topics discussed in the text. These questions should reflect the depth and complexity of the material, similar to what a professional-grade teacher would ask.

        For each question:
        1. Provide the **correct answer** at the end of the question.
        2. **Explain why** the answer is correct, with a focus on the key concepts behind the question.
        3. Specify **which topic** the question is related to, such as "Topic: [Topic Name]."
        4. Ensure the questions are well-structured, engaging, and relevant to the content.
        5. Include at least **one resource or website** where users can learn more about each topic.

        Here is the content you need to generate questions from:

        ======== PDF Content ========
        {concatenated_text}
        =============================

        Generated questions:
        """

        try:
            questions = llm.predict(prompt)  # Use the language model (e.g., Gemini) to generate questions
            insert_chat_log("Generate Questions Request", questions.strip())  # Log the interaction
            return jsonify({"response": questions.strip()}), 200  # Return the generated questions
        except Exception as e:
            logging.error(f"❌ Error generating questions: {str(e)}")
            return jsonify({"response": "⚠️ Failed to generate questions."}), 200
    except Exception as e:
        logging.error(f"❌ Error in generate_questions: {str(e)}")
        return jsonify({"error": "Internal error occurred while generating questions."}), 500
    
    


    
    
def get_answer_from_file(user_query):
    try:
        global recent_file_uid

        if not recent_file_uid:
            return jsonify({"error": "No recent file uploaded"}), 400

        # Initialize the vector store for similarity search
        vectorstore = SupabaseVectorStore(
            client=supabase,
            embedding=embedding_fn,
            table_name="documents",
            query_name="match_documents"
        )

        # Use the vector store to retrieve relevant documents
        retriever = vectorstore.as_retriever()

        # Create a retrieval chain with the language model (Google Gemini or other)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # Use invoke() instead of run() to get the response
        response = qa_chain.invoke(user_query)

        # Extract the result and source_documents
        result = response.get('result')
        source_documents = response.get('source_documents')

        # If relevant content is found, return the answer
        if result:
            insert_chat_log(user_query, result.strip())  # Log the interaction
            return jsonify({
                "file_uuid": recent_file_uid,
                "response": result.strip(),
                "source_documents": [doc.page_content for doc in source_documents]  # Correct way to access content
            }), 200

        # If no relevant content is found, send the query to the Gemini model for an explanation
        else:
            logging.info(f"No relevant content found for the query: {user_query}")
            return get_explanation_from_api(user_query)

    except Exception as e:
        logging.error(f"❌ Error in get_answer_from_file: {str(e)}")
        return jsonify({"error": "Internal error occurred while fetching the answer from file."}), 500

def get_explanation_from_api(user_query):
    try:
        # Simulate a POST request to the /chat route from within this function
        with current_app.test_client() as client:
            # Send the user query to the /chat route as if it were a user request
            response = client.post('/chat', json={"message": user_query})
            
            # Check if the response was successful
            if response.status_code == 200:
                response_json = response.get_json()
                reply = response_json.get("response", "").strip()
                
                if not reply:
                    return jsonify({"error": "No response from Gemini"}), 500
                
                return jsonify({"response": reply}), 200
            else:
                return jsonify({"error": "Something went wrong with the /chat endpoint."}), 500
    except Exception as e:
        logging.error(f"❌ Error in get_explanation_from_api: {str(e)}")
        return jsonify({"error": "Something went wrong while getting an explanation from the external API."}), 500


# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
