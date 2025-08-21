from flask import Blueprint, request, jsonify, make_response
import hashlib
import os
import logging
from datetime import date
from process_pdf_for_quiz import process_pdf_for_quiz
from QA_ANSWER import generate_and_save_mcqs
from matching_q_a import evaluate_and_save_quiz
from utils.ai_helpers import get_summary, get_answer_from_file, generate_questions
import re
from utils.database_helpers import insert_chat_log_supabase_with_conversation
import uuid
import config

quiz_bp = Blueprint('quiz_bp', __name__)

from app import supabase, GEMINI_API_KEY, app, ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# This is an endpoint that seems to have been moved here from app.py
@quiz_bp.route('/quiz-question', methods=['POST'])
def quiz_question():
    user_id = request.form.get("user_id")
    file = request.files.get("file")

    if not user_id or not file:
        return jsonify({"error": "Missing user_id or file."}), 400
    
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)
    
    if len(file_bytes) > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": "File is too large. Max size is 5MB."}), 413
    
    try:
        response = supabase.table('topics') \
            .select('topic_id') \
            .eq('user_id', user_id) \
            .eq('hash_file', file_hash) \
            .execute()
        
        if response.data and len(response.data) > 0:
            logging.info(f"Duplicate upload detected for user {user_id} with file hash {file_hash}")
            return jsonify({"message": "You have already uploaded this file earlier."}), 409
    except Exception as e:
        logging.error(f"Error querying Supabase: {e}")
        return jsonify({"error": "Internal server error checking uploads."}), 500

    if allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        temp_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)

        result = process_pdf_for_quiz(temp_path, GEMINI_API_KEY, user_id, supabase, file_hash)

        if os.path.exists(temp_path):
            os.remove(temp_path)
            logging.info(f"🗑️ Deleted temporary file: {temp_path}")

        if result and result.get("status") == "success":
            return jsonify({"message": "Topics saved successfully."}), 200
        else:
            return jsonify({"error": "Failed to process PDF."}), 500

    return jsonify({"error": "Invalid file type."}), 400

@quiz_bp.route("/generate-questions", methods=['POST', 'OPTIONS'])
def generate_questions_route():
    if request.method == 'OPTIONS':
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
        questions = generate_and_save_mcqs(topic_id, GEMINI_API_KEY, difficulty, user_id)

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
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
        return response, 500

@quiz_bp.route("/submit-answers", methods=["POST", "OPTIONS"])
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

@quiz_bp.route("/api/update-weak-topics", methods=["POST"])
def update_weak_topics():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    today = date.today().isoformat()

    res = supabase.table("user_topic_review_features") \
        .select("topic_id") \
        .eq("user_id", user_id) \
        .lt("next_review_date", today) \
        .execute()

    if not res.data:
        return jsonify({"message": "No topics to update", "updated_count": 0})

    topic_ids = [row["topic_id"] for row in res.data]

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