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
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app, resources={
    r"/chat": {"origins": "http://localhost:5173", "methods": ["POST"]},
    r"/upload": {"origins": "http://localhost:5173", "methods": ["POST"]},
    r"/ask": {"origins": "http://localhost:5173", "methods": ["POST"]},
})

# === Initialize Supabase & Langchain Models ===
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
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

        reply = chat_model.predict(user_message)
        insert_chat_log(user_message, reply, user_id)

        return jsonify({"response": reply}), 200

    except Exception as e:
        logging.error(f"/chat error: {e}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    global recent_file_uid

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logging.info(f"File saved: {file_path}")

        try:
            success, chunk_count, uploaded_filename, file_uid = process_pdf(file_path, supabase, GEMINI_API_KEY)
            recent_file_uid = file_uid
            if success:
                return jsonify({"message": f"PDF processed. {chunk_count} chunks saved."}), 200
            else:
                return jsonify({"error": f"Failed to process PDF: {chunk_count}"}), 500
        except Exception as e:
            logging.error(f"Error during PDF processing: {e}")
            return jsonify({"error": "Failed to process the PDF file."}), 500

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_message = request.json.get("message", "")

        if any(keyword in user_message.lower() for keyword in summary_keywords()):
            return get_summary()
        elif any(keyword in user_message.lower() for keyword in question_keywords()):
            return generate_questions()
        else:
            return get_answer_from_file(user_message)

    except Exception as e:
        logging.error(f"/ask error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# === Intent Detection Keyword Lists ===
def summary_keywords():
    return ["summarize", "overview", "summary", "what is this file about", "give me a summary"]

def question_keywords():
    return ["generate questions", "create questions", "make questions", "quiz questions"]

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
        You are an expert assistant that summarizes documents. Based on the following content, summarize the key points:

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

        text = "\n\n".join(doc['content'] for doc in docs)
        prompt = f"""
        You are a question generator. Create quiz questions based on this:

        ========
        {text}
        ========

        Questions:
        """
        questions = llm.predict(prompt)
        insert_chat_log("Generate Questions Request", questions.strip())
        return jsonify({"response": questions.strip()}), 200
    except Exception as e:
        logging.error(f"Error in generate_questions: {e}")
        return jsonify({"error": "Failed to generate questions"}), 500

def get_answer_from_file(user_query):
    global recent_file_uid
    try:
        if not recent_file_uid:
            return jsonify({"error": "No recent file uploaded"}), 400

        retriever = SupabaseVectorStore(
            client=supabase,
            embedding=embedding_fn,
            table_name="documents",
            query_name="match_documents"
        ).as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        response = qa_chain.invoke(user_query)
        result = response.get('result')
        sources = [doc.page_content for doc in response.get('source_documents', [])]

        if result:
            insert_chat_log(user_query, result.strip())
            return jsonify({"response": result.strip(), "file_uuid": recent_file_uid, "source_documents": sources}), 200
        else:
            return get_explanation_from_api(user_query)
    except Exception as e:
        logging.error(f"Error in get_answer_from_file: {e}")
        return jsonify({"error": "Failed to fetch answer"}), 500

def get_explanation_from_api(user_query):
    try:
        reply = chat_model.predict(user_query)
        return jsonify({"response": reply.strip()}), 200
    except Exception as e:
        logging.error(f"Error in get_explanation_from_api: {e}")
        return jsonify({"error": "Fallback failed"}), 500

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)