import os
import uuid
import logging
from app import supabase
from flask import Flask, jsonify, request
from process_pdf_for_quiz import process_pdf_for_quiz
from QA_ANSWER import generate_and_save_mcqs
from matching_q_a import evaluate_and_save_quiz

def process_quiz_file(file, user_id, file_hash):
    # Save file temporarily
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join('uploads', temp_filename)
    file.save(temp_path)

    # Process PDF for quiz topics and save them to Supabase
    result = process_pdf_for_quiz(temp_path, os.getenv("GEMINI_API_KEY"), user_id, supabase, file_hash)

    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
        logging.info(f"🗑️ Deleted temporary file: {temp_path}")

    if result and result.get("status") == "success":
        return jsonify({"message": "Topics saved successfully."}), 200
    else:
        return jsonify({"error": "Failed to process PDF."}), 500

def generate_quiz_questions(user_id, topic_id, difficulty):
    try:
        questions = generate_and_save_mcqs(topic_id, os.getenv("GEMINI_API_KEY"), difficulty, user_id)

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
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
        return response, 500

def evaluate_submitted_answers(user_id, topic_id, submitted_answers, email_id):
    return evaluate_and_save_quiz(user_id, topic_id, submitted_answers, email_id)