from flask import Flask
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv
from supabase import create_client
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
from mailer import init_mail
from langchain_huggingface import HuggingFaceEmbeddings

# Import route blueprints
from routes.chat_routes import chat_bp
from routes.file_routes import file_bp
from routes.quiz_routes import quiz_bp
from routes.progress_routes import progress_bp
from routes.conversation_routes import conversation_bp

# === Load Environment Variables ===
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bhrwvazkvsebdxstdcow.supabase.co/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_APP_PASSWORD")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

# === Initialize Logging ===
logging.basicConfig(level=logging.DEBUG)

# === Initialize Flask App ===
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CORS(app, resources={
#     r"/chat": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
#     r"/upload": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
#     r"/ask": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
#     r"/quiz-question": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
#     r"/generate-questions": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
#     r"/api/generate_flashcards": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
#     r"/submit-answers": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
#     r"/api/progress/.*": {"origins": "http://localhost:5173", "methods": ["GET", "OPTIONS"]},
#     r"/api/answer-analysis": {"origins": "http://localhost:5173", "methods": ["GET", "OPTIONS"]},
#     r"/api/update-weak-topics": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
#     r"/*": {"origins": "*"}
# }, supports_credentials=True)

# === Dynamic CORS Configuration ===
CORS_ORIGIN = os.getenv("CORS_ALLOW_ORIGIN", "http://localhost:5173")
CORS(app, resources={r"/*": {"origins": CORS_ORIGIN}}, supports_credentials=True)


# Setup Flask-Mail
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=GMAIL_USER,
    MAIL_PASSWORD=GMAIL_PASS,
    MAIL_DEFAULT_SENDER=GMAIL_USER,
)

init_mail(app)

# === Initialize Supabase & Langchain Models ===
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7)
llm1=ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY, temperature=0.3)
memory = ConversationBufferWindowMemory(k=10, return_messages=True)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
# embedding_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
embedding_fn=HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# === Global Variable ===
recent_file_uid = None

# === Helper Functions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Make shared resources available to blueprints
app.config['supabase'] = supabase
app.config['llm'] = llm
app.config['llm1'] = llm1
app.config['conversation'] = conversation
app.config['embedding_fn'] = embedding_fn
app.config['GEMINI_API_KEY'] = GEMINI_API_KEY
app.config['recent_file_uid'] = recent_file_uid
app.config['allowed_file'] = allowed_file

# Register blueprints
app.register_blueprint(chat_bp)
app.register_blueprint(file_bp)
app.register_blueprint(quiz_bp)
app.register_blueprint(progress_bp)
app.register_blueprint(conversation_bp)


@app.route("/health")
def health():
    return {"status": "ok"}


# === Run App ===
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)