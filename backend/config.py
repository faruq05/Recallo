import os
from dotenv import load_dotenv

load_dotenv()

# General
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Flask-Mail (if applicable)
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USERNAME = os.getenv("GMAIL_USER")
MAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
MAIL_DEFAULT_SENDER = os.getenv("GMAIL_USER")