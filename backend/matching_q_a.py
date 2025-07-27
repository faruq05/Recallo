import uuid
import os 
import logging
import json
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client

# Initialize Supabase client (make sure these env variables are set)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bhrwvazkvsebdxstdcow.supabase.co/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def generate_uuid():
    return str(uuid.uuid4())

def evaluate_and_save_quiz(user_id, topic_id, submitted_answers):
    import json

    # Extract question IDs
    question_ids = [ans["question_id"] for ans in submitted_answers]

    # Fetch correct answers and options
    correct_answers_res = supabase.table("quiz_questions") \
        .select("question_id, answer, answer_option_text") \
        .in_("question_id", question_ids) \
        .execute()

    if not correct_answers_res.data:
        raise Exception("Failed to fetch correct answers from Supabase")

    correct_answers_map = {}
    option_text_map = {}

    # Parse correct answers and answer text options
    for q in correct_answers_res.data:
        correct_answers_map[q["question_id"]] = q["answer"]

        try:
            option_text_map[q["question_id"]] = json.loads(q.get("answer_option_text", "{}"))
        except Exception as e:
            logging.warning(f"❌ Failed to parse answer_option_text for question {q['question_id']}: {e}")
            option_text_map[q["question_id"]] = {}

    # Compare answers and prepare for saving
    total_questions = len(submitted_answers)
    correct_count = 0
    answers_to_save = []

    for ans in submitted_answers:
        qid = ans["question_id"]
        selected = ans["selected_answer"]
        correct = correct_answers_map.get(qid)

        option_texts = option_text_map.get(qid, {})  # ✅ Make sure this is above both .get() calls

        correct_text = option_texts.get(correct, "N/A")
        selected_text = option_texts.get(selected, "N/A")

        is_correct = (selected == correct)

        if is_correct:
            correct_count += 1

        answers_to_save.append({
            "answer_id": generate_uuid(),
            "question_id": qid,
            "selected_answer": selected,
            "selected_answer_text": selected_text,
            "is_correct": is_correct
        })

    # Score calculation
    score = (correct_count / total_questions) * 10
    attempt_id = generate_uuid()

    # Save quiz attempt
    attempt_res = supabase.table("quiz_attempts").insert({
        "attempt_id": attempt_id,
        "user_id": user_id,
        "topic_id": topic_id,
        "score": score,
        "submitted_at": datetime.now().isoformat()
    }).execute()

    if not attempt_res.data:
        raise Exception("Failed to insert quiz attempt into Supabase")

    # Add attempt ID to each answer record
    for answer in answers_to_save:
        answer["attempt_id"] = attempt_id

    # Save all answers
    answers_res = supabase.table("quiz_answers").insert(answers_to_save).execute()
    if not answers_res.data:
        raise Exception("Failed to insert quiz answers into Supabase")

    # Update or insert progress
    progress_res = supabase.table("user_topic_progress") \
        .select("*") \
        .eq("user_id", user_id) \
        .eq("topic_id", topic_id) \
        .maybe_single() \
        .execute()

    progress_data = getattr(progress_res, 'data', None)

    if not progress_data:
        insert_progress_res = supabase.table("user_topic_progress").insert({
            "user_id": user_id,
            "topic_id": topic_id,
            "last_score": score,
            "attempts_count": 1,
            "mastered": score >= 7,
            "last_attempt": datetime.now().isoformat()
        }).execute()

        if not insert_progress_res or not insert_progress_res.data:
            raise Exception("Failed to insert user topic progress into Supabase")
    else:
        attempts_count = progress_data.get("attempts_count", 0) + 1
        mastered = progress_data.get("mastered", False) or (score >= 7)

        update_res = supabase.table("user_topic_progress").update({
            "last_score": score,
            "attempts_count": attempts_count,
            "mastered": mastered,
            "last_attempt": datetime.now().isoformat()
        }).eq("user_id", user_id).eq("topic_id", topic_id).execute()

        if not update_res or not update_res.data:
            raise Exception("Failed to update user topic progress in Supabase")

    # Update topic status
    new_status = "Completed" if score > 7 else "Weak"
    status_update = supabase.table("topics").update({
        "topic_status": new_status
    }).eq("topic_id", topic_id).execute()

    if not status_update or not status_update.data:
        logging.warning(f"⚠️ Failed to update topic_status to '{new_status}' for topic {topic_id}")

    return {
        "score": score,
        "total_questions": total_questions,
        "correct_answers": correct_count
    }
		