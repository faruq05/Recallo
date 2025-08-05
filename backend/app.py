from flask import Flask, request, jsonify, session
from flask_cors import CORS, cross_origin
from flask import make_response
import json
import os
import logging
import re
import uuid
import hashlib
import random
from datetime import datetime 
from datetime import date
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
from fetch_text_supabase import fetch_chunk_text_from_supabase
from langchain.schema import HumanMessage
from process_pdf_for_quiz import process_pdf_for_quiz
from QA_ANSWER import generate_and_save_mcqs
from matching_q_a import evaluate_and_save_quiz
from mailer import  init_mail 


# === Load Environment Variables ===
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bhrwvazkvsebdxstdcow.supabase.co/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

CORS(app, resources={
    r"/chat": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
    r"/upload": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
    r"/ask": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
    r"/quiz-question": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
    r"/generate-questions": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
    r"/api/generate_flashcards": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
    r"/submit-answers": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
    r"/api/progress/.*": {"origins": "http://localhost:5173", "methods": ["GET", "OPTIONS"]},
    r"/api/answer-analysis": {"origins": "http://localhost:5173", "methods": ["GET", "OPTIONS"]},
    r"/api/update-weak-topics": {"origins": "http://localhost:5173", "methods": ["POST", "OPTIONS"]},
    r"/*": {"origins": "*"}
    
}, supports_credentials=True)

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

# --- Supabase Insert Helper ---
def insert_chat_log_supabase_with_conversation(user_id, conv_id, user_msg, resp_msg):
    """Insert chat log with conversation tracking"""
    try:
        message_id = str(uuid.uuid4())
        data = {
            "user_id": user_id,
            "conversation_id": conv_id,
            "user_message": user_msg,
            "response_message": resp_msg,
            "created_at": datetime.now().isoformat(),
            "message_id": message_id
        }
        supabase.table("chat_logs").insert(data).execute()

        # Update conversation timestamp
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("conversation_id", conv_id).execute()

        logging.info(f"Inserted chat log into Supabase for conversation {conv_id}")
    except Exception as e:
        logging.error(f"Supabase insert error: {e}")

# === Endpoints ===
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message", "")
        user_id = request.json.get("user_id")
        conv_id = request.json.get("conversation_id")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        if not user_id:
            return jsonify({"error": "No user ID provided"}), 400
        
        logging.info(f"Received message: {user_message}")
        
        # Step 1: Handle conversation ID safely
        if not conv_id:
            conv_id = str(uuid.uuid4())
            supabase.table("conversations").insert({
                "conversation_id": conv_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "title": "New Chat"
            }).execute()

        # Enhanced educational and conversational prompt
        prompt = f"""
        You are an AI assistant with a deep knowledge base designed to help users learn and understand any topic in the most effective and engaging way. Your role is to provide clear, accurate, and detailed explanations, making complex topics easy to understand. 

        Respond to the user with a friendly, conversational tone, as if you're explaining the concept to a student. Break down the topic step by step when necessary, and give real-life examples to aid comprehension. Also, offer YouTube video suggestions that are relevant to the topic for further learning.

        Be empathetic, patient, and provide well-rounded answers. If the user asks for clarifications or examples, be ready to offer more detailed responses and give helpful suggestions.

        User message: {user_message}
        """
        
        
        # Run the conversation chain with memory
        ai_reply = conversation.predict(input=user_message)
        
        # Step 3: Insert log
        insert_chat_log_supabase_with_conversation(user_id, conv_id, user_message, ai_reply)

        # Step 4: If this is the first message, generate smart title
        log_check = supabase.table("chat_logs").select("id").eq("conversation_id", conv_id).execute()
        if log_check.data and len(log_check.data) == 1:
            smart_title = generate_title(user_message, ai_reply)
            supabase.table("conversations").update({
                "title": smart_title
            }).eq("conversation_id", conv_id).execute()


        return jsonify({
            "response": ai_reply,
            "conversation_id": conv_id
        }), 200


    except Exception as e:
        logging.error(f"/chat error: {e}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    global recent_file_uid 
    logging.info("Upload route hit")
    user_id = request.form.get("user_id")  # Get user ID from the form data
    print(f"User ID from upload: {user_id}")

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read file bytes to compute hash
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)  # Reset file pointer after reading

    # Check file size (adjust if needed since content_length might not always be available)
    if file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": "File is too large. Max size is 5MB"}), 413

    # Check if file hash already exists for this user in Supabase
    try:
        response = supabase.table('documents') \
            .select('id') \
            .eq('user_id', user_id) \
            .eq('hash_file', file_hash) \
            .execute()
        
        if response.data and len(response.data) > 0:
            # File already uploaded by this user
            logging.info(f"Duplicate upload detected for user {user_id} with file hash {file_hash}")
            return jsonify({"message": "You have already uploaded this file earlier."}), 409
    except Exception as e:
        logging.error(f"Error querying Supabase: {e}")
        return jsonify({"error": "Internal server error checking uploads."}), 500

    # Validate file extension/type
    if file and allowed_file(file.filename):
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        logging.info(f"📥 File saved: {file_path}")

        try:
            # Process your PDF file as usual
            success, chunk_count, uploaded_filename, file_uid = process_pdf(file_path, supabase, GEMINI_API_KEY, user_id,file_hash)

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



# --- Endpoint to update weak topics ---
@app.route("/api/update-weak-topics", methods=["POST"])
def update_weak_topics():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    today = date.today().isoformat()

    # Step 1: Find overdue topics
    res = supabase.table("user_topic_review_features") \
        .select("topic_id") \
        .eq("user_id", user_id) \
        .lt("next_review_date", today) \
        .execute()

    if not res.data:
        return jsonify({"message": "No topics to update", "updated_count": 0})

    topic_ids = [row["topic_id"] for row in res.data]

    # Step 2: Update topic_status in 'topics' table
    for topic_id in topic_ids:
        supabase.table("topics") \
            .update({"topic_status": "Weak"}) \
            .eq("user_id", user_id) \
            .eq("topic_id", topic_id) \
            .execute()
            
    attempt_res = supabase.table("quiz_attempts") \
            .select("attempt_id") \
            .eq("user_id", user_id) \
            .eq("topic_id", topic_id) \
            .order("submitted_at", desc=True) \
            .limit(1) \
            .execute()

    if attempt_res.data:
            attempt_id = attempt_res.data[0]["attempt_id"]
            supabase.table("quiz_attempts") \
                .update({"score": 0}) \
                .eq("attempt_id", attempt_id) \
                .execute()

    return jsonify({
        "message": f"Updated {len(topic_ids)} topics to 'Weak'.",
        "updated_count": len(topic_ids),
        "topic_ids": topic_ids
    })



# topic jsx route
@app.route('/quiz-question', methods=['POST'])
def quiz_question():
    user_id = request.form.get("user_id")
    file = request.files.get("file")

    if not user_id or not file:
        return jsonify({"error": "Missing user_id or file."}), 400
    
    # Read file bytes to compute hash
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)  # Reset file pointer after reading

    if len(file_bytes) > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": "File is too large. Max size is 5MB."}), 413
    
    
    # Check if file hash already exists for this user in Supabase
    try:
        response = supabase.table('topics') \
            .select('topic_id') \
            .eq('user_id', user_id) \
            .eq('hash_file', file_hash) \
            .execute()
        
        if response.data and len(response.data) > 0:
            # File already uploaded by this user
            logging.info(f"Duplicate upload detected for user {user_id} with file hash {file_hash}")
            return jsonify({"message": "You have already uploaded this file earlier."}), 409
    except Exception as e:
        logging.error(f"Error querying Supabase: {e}")
        return jsonify({"error": "Internal server error checking uploads."}), 500

    if  allowed_file(file.filename):
        # Save file temporarily
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        temp_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)

        # Process PDF for quiz topics and save them to Supabase
        result = process_pdf_for_quiz(temp_path, GEMINI_API_KEY, user_id, supabase,file_hash)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logging.info(f"🗑️ Deleted temporary file: {temp_path}")

        if result and result.get("status") == "success":
            return jsonify({"message": "Topics saved successfully."}), 200
        else:
            return jsonify({"error": "Failed to process PDF."}), 500

    return jsonify({"error": "Invalid file type."}), 400

# exam route
@app.route("/generate-questions", methods=['POST', 'OPTIONS'])
def generate_questions():
    if request.method == 'OPTIONS':
        # Respond to preflight CORS request
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 204

    data = request.get_json()
    print("the data is ", data)
    user_id = data.get("user_id")
    topic_id = data.get("topic_id")
    difficulty = data.get("difficulty_mode", "hard")

    try:
        questions = generate_and_save_mcqs(topic_id, GEMINI_API_KEY, difficulty,user_id)

        # Strip correct_answer for frontend
        questions_for_frontend = [
            {   
                "question_id": q["question_id"],  
                "question_text": q["question_text"],
                "options": q["options"],
                "correct_answer": q.get("correct_answer"),
                "answer_text": q.get("answer_text"), 
                "explanation": q.get("explanation")
            }
            for q in questions
        ]

        response = jsonify({"questions": questions_for_frontend})
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
        return response, 200
    except Exception as e:
        import traceback
        traceback.print_exc()  # <-- logs to terminal
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
        return response, 500

# submit answer
@app.route("/submit-answers", methods=["POST", "OPTIONS"])
@cross_origin()  # Add this decorator
def submit_answers():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON payload"}), 400

        user_id = data.get("user_id")
        topic_id = data.get("topic_id")
        email_id = data.get("email")
        submitted_answers = data.get("submitted_answers")
        

        if not user_id or not topic_id or not submitted_answers:
            return jsonify({"error": "Missing one or more required fields: user_id, topic_id, submitted_answers"}), 400

        # Validate each answer object has a 'question_id' and 'selected_answer'
        for ans in submitted_answers:
            if not isinstance(ans, dict):
                return jsonify({"error": "Invalid answer format. Each answer must be an object."}), 400
            if "question_id" not in ans or "selected_answer" not in ans:
                return jsonify({"error": "Each answer must include 'question_id' and 'selected_answer'"}), 400

        result = evaluate_and_save_quiz(user_id, topic_id, submitted_answers, email_id)

        return jsonify({"message": "Quiz submitted successfully", "result": result}), 200

    except Exception as e:
        logging.exception("Error while processing submitted answers")
        return jsonify({"error": str(e)}), 500

# answer analysis in progress jsx   
@app.route("/api/answer-analysis")
def get_answer_analysis():
    topic_id = request.args.get("topic_id")
    user_id = request.args.get("user_id")
    attempt_number = request.args.get("attempt_number")

    if not (topic_id and user_id and attempt_number):
        return jsonify({"error": "Missing required parameters"}), 400

    # Get all attempts for this user and topic
    attempts_resp = supabase.table("quiz_attempts") \
        .select("attempt_id, submitted_at, score") \
        .eq("topic_id", topic_id) \
        .eq("user_id", user_id) \
        .order("submitted_at") \
        .execute()
    attempts = attempts_resp.data

    if not attempts:
        return jsonify({"error": "No attempts found"}), 404

    try:
        attempt_idx = int(attempt_number) - 1
        selected_attempt = attempts[attempt_idx]
    except (IndexError, ValueError):
        return jsonify({"error": "Invalid attempt number"}), 400

    attempt_id = selected_attempt["attempt_id"]

    # Fetch answers for this attempt (including selected_answer_text)
    answers_resp = supabase.table("quiz_answers") \
        .select("question_id, selected_answer, selected_answer_text, is_correct") \
        .eq("attempt_id", attempt_id) \
        .execute()
    answers = answers_resp.data or []

    question_ids = [a["question_id"] for a in answers]
    if not question_ids:
        return jsonify({"error": "No answers found for this attempt"}), 404

    # Fetch questions info
    questions_resp = supabase.table("quiz_questions") \
        .select("question_id, prompt, answer, answer_option_text, explanation") \
        .in_("question_id", question_ids) \
        .execute()
    questions = questions_resp.data or []

    analysis = []

    for answer in answers:
        q = next((item for item in questions if item["question_id"] == answer["question_id"]), None)
        if not q:
            continue

        # Initialize defaults
        correct_option = q["answer"]
        selected_option = answer["selected_answer"]
        correct_option_text = ""
        selected_option_text = answer.get("selected_answer_text", "")
        options = {}

        try:
            if q["answer_option_text"]:
                options = json.loads(q["answer_option_text"])
                # Get correct option text from the JSON
                correct_option_text = options.get(correct_option, "")
                
                # If selected_answer_text wasn't stored, get it from options
                if not selected_option_text and selected_option:
                    selected_option_text = options.get(selected_option, "")

        except Exception as e:
            print(f"Error processing options for question {q['question_id']}: {str(e)}")

        analysis.append({
            "question_id": q["question_id"],
            "question_text": q["prompt"],
            "correct_option": correct_option,
            "correct_option_text": correct_option_text,
            "selected_option": selected_option,
            "selected_option_text": selected_option_text,
            "explanation": q.get("explanation", "No explanation provided"),
            "is_correct": answer["is_correct"],
            "all_options": options,
        })

    return jsonify({
        "questions": analysis,
        "attempt_data": {  # Add attempt metadata
            "score": selected_attempt.get("score"),
            "submitted_at": selected_attempt.get("submitted_at")
        }
    })
    
    
# ===== FLASHCARD GENERATION ===== #
@app.route("/api/generate_flashcards", methods=["POST"])
def generate_flashcards():
    try:
        data = request.get_json()
        attempt_id = data.get("attempt_id")
        user_id = data.get("user_id")
        topic_id = data.get("topic_id")
        print(f"Attempt ID: {attempt_id}, User ID: {user_id}")

        if not attempt_id or not user_id:
            return jsonify({"error": "Missing attempt_id or user_id"}), 400
        
        
        existing = supabase.from_("flashcards") \
            .select("core_concept, key_theory, common_mistake") \
            .eq("user_id", user_id) \
            .eq("attempt_id", attempt_id) \
            .execute()

        if existing.data and len(existing.data) == 10:
            return jsonify({
                "flashcards": existing.data,
                "message": "fetched",
            })
        
        
        # Step 1: Fetch quiz answers for this attempt
        answers_response = supabase.from_("quiz_answers").select(
            "question_id, selected_answer, is_correct"
        ).eq("attempt_id", attempt_id).execute()

        quiz_answers = answers_response.data or []
        
        if not quiz_answers:
            return jsonify({"error": "No answers found for this attempt"}), 404
        
        question_ids = [qa["question_id"] for qa in quiz_answers]
        
        # Step 2: Fetch corresponding questions
        questions_response = supabase.from_("quiz_questions").select(
            "question_id, prompt, answer, explanation, answer_option_text, concept_id"
        ).in_("question_id", question_ids).execute()

        question_map = {q["question_id"]: q for q in questions_response.data}
        
        # Fetch merged content from the topic
        topic_response = supabase.from_("topics").select("merged_content").eq("topic_id", topic_id).single().execute()
        merged_content = topic_response.data.get("merged_content") if topic_response.data else None

        
        merged = []
        for qa in quiz_answers:
            q = question_map.get(qa["question_id"])
            if q:
                merged.append({
                    "question_id": qa["question_id"],
                    "prompt": q["prompt"],
                    "options": q["answer_option_text"],
                    "correct_answer": q["answer"],
                    "selected_answer": qa["selected_answer"],
                    "is_correct": qa["is_correct"]
                })
                
        if not merged:
            return jsonify({"error": "No matching questions for answers"}), 404
                
        # --- Step 4: Sample 8 incorrect + (2 correct or more) ---
        incorrect = [item for item in merged if item["is_correct"] is False]
        correct = [item for item in merged if item["is_correct"] is True]

        random.shuffle(incorrect)
        random.shuffle(correct)

        incorrect_sample = incorrect[:min(8, len(incorrect))]
        remaining = 10 - len(incorrect_sample)
        correct_sample = correct[:min(remaining, len(correct))]

        flashcards_base = incorrect_sample + correct_sample
        random.shuffle(flashcards_base)
        
        
        # --- Step 5: Format examples for LLM prompt ---
        examples = [
            {
                "question": item["prompt"],
                "correct_answer": item["correct_answer"],
                "user_answer": item["selected_answer"],
                "is_wrong": item["is_correct"] is False
            }
            for item in flashcards_base
        ]

        prompt_examples = "\n".join(
            f"Q: {ex['question']}\nA: {ex['correct_answer']}\n"
            f"{'User Mistake: ' + (ex['user_answer'] or 'N/A') if ex['is_wrong'] else ''}"
            for ex in examples
        )



        # System prompt
        system_prompt = """You are a study assistant that generates concept flashcards from quiz questions.

            Each flashcard must contain:

            1. "core_concept" — the fundamental concept
            2. "key_theory" — a clear explanation of the concept
            3. "common_mistake" — (only if user got it wrong)

            🔁 You MUST return **exactly 10 items** as a JSON array.

            💡 Format:
            [
            {
                "core_concept": "...",
                "key_theory": "...",
                "common_mistake": "..." // only if relevant
            },
            ...
            ]

            ⛔️ Do not include any markdown, headings, or extra text.
            ⛔️ Do not repeat items. Stop after 10.
            ✅ Output should start with `[` and end with `]`.
            """



        # final_prompt = f"{system_prompt}\n\nQuiz Questions:\n{prompt_examples}"
        final_prompt = f"{system_prompt}\n\n📚 Topic Context:\n{merged_content or 'N/A'}\n\nQuiz Questions:\n{prompt_examples}"

        
        
        # --- Step 6: Log for debugging ---
        print("\n🧪 Sampled Flashcard Data:")
        for ex in examples:
            print(json.dumps(ex, indent=2))

        print("\n🧠 Final Prompt Sent to LLM:")
        print(final_prompt)

        # --- Step 7: LLM Call ---
        llm_response = llm.invoke([HumanMessage(content=final_prompt)])
        raw_text = llm_response.content

        try:
            flashcards = json.loads(raw_text)
        except json.JSONDecodeError:
            print("\n❌ LLM response parsing failed:")
            print(raw_text)
            return jsonify({
                "error": "Failed to parse flashcards",
                "llm_response": raw_text
            }), 500

        if (
            not isinstance(flashcards, list)
            or len(flashcards) != 10
            or not all("core_concept" in fc and "key_theory" in fc for fc in flashcards)
        ):
            raise ValueError("Flashcards must contain core_concept and key_theory, 10 items")

        print("\n✅ Final Flashcards:")
        print(json.dumps(flashcards, indent=2))

        # ✅ Step 4: Save flashcards
        now = datetime.now().isoformat()
        records = [{
            "user_id": user_id,
            "attempt_id": attempt_id,
            "topic_id": topic_id,
            "core_concept": fc["core_concept"],
            "key_theory": fc["key_theory"],
            "common_mistake": fc.get("common_mistake"),
            "created_at": now
        } for fc in flashcards]

        insert = supabase.from_("flashcards").insert(records).execute()
        if not insert.data:
            raise Exception("Failed to save flashcards to Supabase")

        return jsonify({"flashcards": flashcards,"message": "Generated"}), 200

    except Exception as e:
        logging.exception("Flashcard generation failed")
        return jsonify({"error": str(e)}), 500

    

# Helper to convert 'A', 'B' etc. to option text
def option_letter_to_text(letter, answer_option_text):
    if not letter or not answer_option_text:
        return ""

    options = {}
    for line in answer_option_text.strip().splitlines():
        if "." in line:
            parts = line.strip().split(".", 1)
            if len(parts) == 2:
                key = parts[0].strip().upper()
                val = parts[1].strip()
                options[key] = val

    return options.get(letter.upper(), "")

# chat mode route
@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_message = request.json.get("message", "")
        user_id = request.json.get("user_id")
        conv_id = request.json.get("conversation_id")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        if not user_id:
            return jsonify({"error": "No user_id provided"}), 400

        # Validate or create conversation
        if conv_id:
            existing_conv = supabase.table("conversations").select("conversation_id") \
                .eq("conversation_id", conv_id).eq("user_id", user_id).execute()
            if not existing_conv.data:
                conv_id = None

        # Step 2: Create new conversation with placeholder title
        is_new_conversation = False
        if not conv_id:
            conv_id = str(uuid.uuid4())
            supabase.table("conversations").insert({
                "conversation_id": conv_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "title": "New Chat"
            }).execute()
            is_new_conversation = True
        
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

        # Step 4: Route logic
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', user_message, re.IGNORECASE) for keyword in summary_keywords):
            summary_response, status_code = get_summary(user_id)
            response_text = summary_response.get_json()["response"]

        elif any(re.search(r'\b' + re.escape(keyword) + r'\b', user_message, re.IGNORECASE) for keyword in question_keywords):
            summary_response, status_code = generate_questions(user_id)
            response_text = summary_response.get_json()["response"]

        else:
            summary_response, status_code = get_answer_from_file(user_message, user_id)
            response_text = summary_response.get_json()["response"]

        # Step 5: Save to chat log
        insert_chat_log_supabase_with_conversation(user_id, conv_id, user_message, response_text)

        # Step 6: Generate title if this is first message
        if is_new_conversation:
            smart_title = generate_title(user_message, response_text)
            supabase.table("conversations").update({
                "title": smart_title
            }).eq("conversation_id", conv_id).execute()

        # Step 7: Return
        return jsonify({"response": response_text, "conversation_id": conv_id}), status_code


    except Exception as e:
        logging.error(f"❌ Ask route error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
# === Functional Chains ===
def get_summary(user_id):
    try:
        print(f"Generating summary for user: {user_id}")

        # Step 1: Get the most recently uploaded file for this user
        response = supabase.table('documents') \
            .select('file_uuid') \
            .eq('user_id', user_id) \
            .order('uploaded_at', desc=True) \
            .limit(1) \
            .execute()

        if not response.data:
            return jsonify({"error": "No recent file uploaded."}), 400

        file_uuid = response.data[0]['file_uuid']
        print("📄 Using file UUID:", file_uuid)

        # Step 2: Get all chunks for that file
        chunks_res = supabase.table('documents') \
            .select('content') \
            .eq('file_uuid', file_uuid) \
            .execute()

        docs = chunks_res.data
        if not docs:
            return jsonify({"error": "No content found for this file."}), 404

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


        # Optional: log the interaction
        # insert_chat_log("Summarize Request", summary)

        return jsonify({"response": summary}), 200

    except Exception as e:
        logging.error(f"Error in get_summary: {e}")
        return jsonify({"error": "Failed to summarize"}), 500

def generate_questions(user_id):
    try:
        print(f"📌 Generating questions for user: {user_id}")

        # Step 1: Get the most recently uploaded file for this user
        response = supabase.table('documents') \
            .select('file_uuid') \
            .eq('user_id', user_id) \
            .order('uploaded_at', desc=True) \
            .limit(1) \
            .execute()

        if not response.data:
            return jsonify({"error": "No recent file uploaded."}), 400

        file_uuid = response.data[0]['file_uuid']
        print(f"✅ Using file UUID: {file_uuid}")

        # Step 2: Get all chunks for that file
        docs_response = supabase.table('documents') \
            .select('content') \
            .eq('file_uuid', file_uuid) \
            .execute()

        docs = docs_response.data
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
        # insert_chat_log("Generate Questions Request", questions.strip())
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
        print("get_answer_from_file called with user_query:", user_query, "and user_id:", user_id)

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
            chunk_id = match['id']
            chunk_data = fetch_chunk_text_from_supabase(supabase, chunk_id, user_id)

            if chunk_data:
                relevant_docs.append(chunk_data)
                print(f"Chunk Text: {chunk_data}")
            

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


    
@app.route("/api/progress/<user_id>", methods=["GET"])
def get_user_progress(user_id):
    try:
        # Fetch all quiz attempts
        attempts_response = supabase.table("quiz_attempts") \
            .select("topic_id, score, submitted_at") \
            .eq("user_id", user_id) \
            .order("submitted_at", desc=True) \
            .execute()

        attempts = attempts_response.data or []

        if not attempts:
            return jsonify([]), 200

        topic_ids = list({a["topic_id"] for a in attempts})

        # Fetch topic metadata
        topics_response = supabase.table("topics") \
            .select("topic_id, title, file_name") \
            .in_("topic_id", topic_ids) \
            .execute()

        topics = topics_response.data or []

        return jsonify({
            "attempts": attempts,
            "topics": topics
        }), 200

    except Exception as e:
        logging.error(f"Error fetching progress: {e}")
        return jsonify({"error": "Failed to fetch progress"}), 500
    
    
    
# --- Conversation Management ---
@app.route("/api/conversations", methods=["GET", "POST"])
def conversations():
    if request.method == "GET":
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        try:
            response = supabase.table("conversations").select(
                "conversation_id, title, created_at, updated_at"
            ).eq("user_id", user_id).order("updated_at", desc=True).execute()
            return jsonify(response.data), 200
        except Exception as e:
            logging.error(f"Error fetching conversations: {e}")
            return jsonify({"error": "Failed to fetch conversations"}), 500

    elif request.method == "POST":
        data = request.get_json()
        user_id = data.get("user_id")
        title = data.get("title", "New Chat")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        conv_id = str(uuid.uuid4())
        try:
            response = supabase.table("conversations").insert({
                "conversation_id": conv_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "title": title
            }).execute()
            return jsonify(response.data[0]), 201
        except Exception as e:
            logging.error(f"Error creating conversation: {e}")
            return jsonify({"error": "Failed to create conversation"}), 500


@app.route("/api/conversations/<conversation_id>", methods=["PUT", "DELETE"])
def update_or_delete_conversation(conversation_id):
    if request.method == "PUT":
        data = request.get_json()
        title = data.get("title")
        if not title:
            return jsonify({"error": "Title is required"}), 400

        try:
            supabase.table("conversations").update({
                "title": title,
                "updated_at": datetime.now().isoformat()
            }).eq("conversation_id", conversation_id).execute()
            return jsonify({"message": "Conversation renamed"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif request.method == "DELETE":
        try:
            supabase.table("chat_logs").delete().eq("conversation_id", conversation_id).execute()
            supabase.table("conversations").delete().eq("conversation_id", conversation_id).execute()
            return jsonify({"message": "Conversation deleted"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/api/conversations/<conv_id>/logs", methods=["GET"])
def get_chat_logs(conv_id):
    try:
        uuid.UUID(conv_id)  # Validate UUID format
        logs = supabase.table("chat_logs").select(
            "user_message, response_message, created_at"
        ).eq("conversation_id", conv_id).order("created_at").execute()
        return jsonify(logs.data), 200
    except Exception as e:
        return jsonify({"error": "Invalid or missing conversation ID"}), 400
    
    
def generate_title(user_message, llm_response):
    try:
        prompt = f"""
        Generate a short and meaningful title (3 to 6 words max) for a conversation based on this exchange:

        User: {user_message}
        Assistant: {llm_response}

        The title should be concise, informative, and not include any quotation marks.
        """
        result = llm.predict(prompt)
        return result.strip().replace('"', '')
    except Exception as e:
        logging.warning(f"⚠️ Failed to generate title: {e}")
        return "New Chat"



# === Run App ===
if __name__ == '__main__':
    app.run(debug=True, port=5000)