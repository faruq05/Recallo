from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from supabase import create_client
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA

from upload_pdf import process_pdf  # ✅ new import


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
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-1.5-pro"
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# === File type check ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Supabase logging ===
def insert_chat_log(user_message, response_message):
    try:
        data = {"user_message": user_message, "response_message": response_message}
        response = supabase.table("chat_logs").insert(data).execute()
        if response.status_code == 201:
            logging.info("Message successfully inserted into Supabase.")
        else:
            logging.warning(f"Supabase insert status: {response.status_code}")
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

        reply = chat_model.predict(user_message)

        if not reply:
            return jsonify({"error": "No response from Gemini"}), 500

        logging.info(f"Generated reply: {reply}")
        insert_chat_log(user_message, reply)

        return jsonify({"response": reply}), 200

    except Exception as e:
        logging.error(f"/chat error: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

# === File Upload Endpoint ===


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logging.info(f"📥 File saved: {file_path}")

        # ✅ Call your separate module to handle chunking + saving
        success, result = process_pdf(file_path, supabase, GEMINI_API_KEY)

        if success:
            return jsonify({
    "message": f"PDF processed. {result} chunks saved.",
    "filename": file.filename
}), 200
        else:
            return jsonify({"error": f"Failed to process PDF: {result}"}), 500

    return jsonify({"error": "Invalid file type"}), 400




# Setup Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

# Embedding model
embedding_fn = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)


# Vector store from Supabase
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embedding_fn,
    table_name="documents",
    query_name="match_documents"
)

# Retriever (RAG engine)
retriever = vectorstore.as_retriever()

# Retrieval-QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Actual endpoint
@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_message = request.json.get("message", "")
        filename = request.json.get("filename", None)
        document_mode = request.json.get("document_mode", False)

        if not user_message:
            return jsonify({"error": "No question provided"}), 400

        if document_mode and not filename:
            return jsonify({"error": "No document selected."}), 400

        if document_mode:
            # 📄 Document mode: retrieve from Supabase by filename
            retriever = vectorstore.as_retriever(
                search_kwargs={"filter": {"filename": filename}}
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            result = qa_chain({"query": user_message})

            for doc in result["source_documents"]:
                print("📄 Retrieved chunk:", doc.page_content[:200], "\nFrom:", doc.metadata)

            return jsonify({"response": result["result"]}), 200

        else:
            # 💬 Chat mode: just use Gemini model directly
            reply = chat_model.predict(user_message)
            return jsonify({"response": reply}), 200

    except Exception as e:
        logging.error(f"Ask error: {str(e)}")
        return jsonify({"error": "Failed to get answer"}), 500
   
# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)