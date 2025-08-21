from flask import Blueprint, request, jsonify
import json
import logging
from utils.helpers import option_letter_to_text
import config

progress_bp = Blueprint('progress_bp', __name__, url_prefix='/api/progress')

from app import supabase

@progress_bp.route("/<user_id>", methods=["GET"])
def get_user_progress(user_id):
    try:
        attempts_response = supabase.table("quiz_attempts") \
            .select("topic_id, score, submitted_at") \
            .eq("user_id", user_id) \
            .order("submitted_at", desc=True) \
            .execute()

        attempts = attempts_response.data or []

        if not attempts:
            return jsonify([]), 200

        topic_ids = list({a["topic_id"] for a in attempts})

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

@progress_bp.route("/api/answer-analysis")
def get_answer_analysis():
    topic_id = request.args.get("topic_id")
    user_id = request.args.get("user_id")
    attempt_number = request.args.get("attempt_number")

    if not (topic_id and user_id and attempt_number):
        return jsonify({"error": "Missing required parameters"}), 400

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

    answers_resp = supabase.table("quiz_answers") \
        .select("question_id, selected_answer, selected_answer_text, is_correct") \
        .eq("attempt_id", attempt_id) \
        .execute()
    answers = answers_resp.data or []

    question_ids = [a["question_id"] for a in answers]
    if not question_ids:
        return jsonify({"error": "No answers found for this attempt"}), 404

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

        correct_option = q["answer"]
        selected_option = answer["selected_answer"]
        correct_option_text = ""
        selected_option_text = answer.get("selected_answer_text", "")
        options = {}

        try:
            if q["answer_option_text"]:
                options = json.loads(q["answer_option_text"])
                correct_option_text = options.get(correct_option, "")
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
        "attempt_data": {
            "score": selected_attempt.get("score"),
            "submitted_at": selected_attempt.get("submitted_at")
        }
    })