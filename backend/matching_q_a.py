import uuid
import os 
import logging
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
    # Extract question IDs
    question_ids = [ans["question_id"] for ans in submitted_answers]

    # Fetch correct answers from Supabase
    correct_answers_res = supabase.table("quiz_questions") \
        .select("question_id, answer") \
        .in_("question_id", question_ids) \
        .execute()

    if not correct_answers_res.data:
        raise Exception("Failed to fetch correct answers from Supabase")

    correct_answers_map = {q["question_id"]: q["answer"] for q in correct_answers_res.data}

    # Compare answers and prepare answers data to save
    total_questions = len(submitted_answers)
    correct_count = 0
    answers_to_save = []

    for ans in submitted_answers:
        qid = ans["question_id"]
        selected = ans["selected_answer"]
        
        correct = correct_answers_map.get(qid)
        is_correct = (selected == correct)
        if is_correct:
            correct_count += 1

        answers_to_save.append({
            "answer_id": generate_uuid(),
            "question_id": qid,
            "selected_answer": selected,
            "is_correct": is_correct
        })

    # Calculate score out of 10
    score = (correct_count / total_questions) * 10

    # Insert quiz attempt
    attempt_id = generate_uuid()
    attempt_res = supabase.table("quiz_attempts").insert({
        "attempt_id": attempt_id,
        "user_id": user_id,
        "topic_id": topic_id,
        "score": score,
        "submitted_at": datetime.now().isoformat()
    }).execute()

    if not attempt_res.data:
        raise Exception("Failed to insert quiz attempt into Supabase")

    # Insert quiz answers linked to the attempt
    for answer in answers_to_save:
        answer["attempt_id"] = attempt_id

    answers_res = supabase.table("quiz_answers").insert(answers_to_save).execute()
    if not answers_res.data:
        raise Exception("Failed to insert quiz answers into Supabase")

    # Update user_topic_progress table
    # Check existing progress
    progress_res = supabase.table("user_topic_progress") \
        .select("*") \
        .eq("user_id", user_id) \
        .eq("topic_id", topic_id) \
        .maybe_single() \
        .execute()
        
    print("progress_res:", progress_res)
    print("progress_res.data:", getattr(progress_res, 'data', None))
    
    if progress_res is None:
    # No progress record found — insert new
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
        # Existing progress found — update record
        attempts_count = progress_res.model_dump().get("attempts_count", 0) + 1
        mastered = progress_res.data.get("mastered", False) or (score >= 7)
        update_res = supabase.table("user_topic_progress").update({
            "last_score": score,
            "attempts_count": attempts_count,
            "mastered": mastered,
            "last_attempt": datetime.now().isoformat()
        }).eq("user_id", user_id).eq("topic_id", topic_id).execute()

        if not update_res or not update_res.data:
            raise Exception("Failed to update user topic progress in Supabase")


    return {
        "score": score,
        "total_questions": total_questions,
        "correct_answers": correct_count
    }
