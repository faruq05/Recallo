from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from google import genai
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

# ✅ Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# API and Supabase credentials (use environment variables or hardcode securely)
VITE_SUPABASE_URL = "https://bhrwvazkvsebdxstdcow.supabase.co/"
VITE_SUPABASE_KEY = os.environ.get('VITE_SUPABASE_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')


# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/chat": {"origins": "http://localhost:5173", "methods": ["POST"]},
    r"/upload": {"origins": "http://localhost:5173", "methods": ["POST"]}})

# File upload config
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Creates folder if not exists

# Initialize Supabase
supabase = create_client(VITE_SUPABASE_URL, VITE_SUPABASE_KEY)

# Initialize Google GenAI (Gemini) client
client = genai.Client(api_key=GEMINI_API_KEY)

# Supabase insertion function
def insert_chat_log(user_message, response_message):
    try:
        data = {"user_message": user_message, "response_message": response_message}
        response = supabase.table("chat_logs").insert(data).execute()
        if response.status_code == 201:
            logging.info("Message successfully inserted into Supabase.")
        else:
            logging.error(f"Error inserting into Supabase: {response.status_code}")
    except Exception as e:
        logging.error(f"Error during Supabase insertion: {str(e)}")

# Google Gemini (GenAI) request function
def genai_query(user_message):
    try:
        # Call the Google Gemini model to generate content
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Or another supported model
            contents=user_message
        )

        # Access the generated text
        return response.text.strip()

    except Exception as e:
        logging.error(f"Error in Gemini request: {str(e)}")
        return None

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        logging.info(f"Received message: {user_message}")

        # Send the message to Google Gemini
        reply = genai_query(user_message)

        if not reply:
            return jsonify({"error": "No response from Google Gemini."}), 500

        logging.info(f"Generated reply: {reply}")

        # Log to Supabase
        insert_chat_log(user_message, reply)

        return jsonify({"response": reply}), 200

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong. Please try again later."}), 500

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
        logging.info(f"Uploaded file saved at: {file_path}")
        return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200

    return jsonify({"error": "Invalid file type"}), 400

# Run app
if __name__ == '__main__':
    app.run(debug=True)