import uuid
import os 
import logging
import json
from dotenv import load_dotenv
from datetime import datetime,timedelta, date
from supabase import create_client
import joblib
import numpy as np
from mailer import send_email

# Initialize Supabase client (make sure these env variables are set)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bhrwvazkvsebdxstdcow.supabase.co/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

model = joblib.load("model.pkl")


def generate_uuid():
    return str(uuid.uuid4())

def evaluate_and_save_quiz(user_id, topic_id, submitted_answers, email_id=None):
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
    
    
    # --- 🔮 MODEL PREDICTION FOR NEXT REVIEW DATE ---
    # Fetch features from user_topic_review_features after trigger updates stats
    review_data = supabase.table("user_topic_review_features") \
    .select("latest_score, avg_score, attempts_count, days_since_last_attempt") \
    .eq("user_id", user_id) \
    .eq("topic_id", topic_id) \
    .maybe_single() \
    .execute()

    if review_data.data:
        latest_score = review_data.data.get("latest_score", score)
        avg_score = review_data.data.get("avg_score", score)
        attempts_count = review_data.data.get("attempts_count", 1)
        days_since_last_attempt = review_data.data.get("days_since_last_attempt", 0)

        X = np.array([[latest_score, avg_score, attempts_count, days_since_last_attempt]])

        predicted_days = int(round(model.predict(X)[0]))
        next_review_date = date.today() + timedelta(days=predicted_days)

        print(f"Predicted days: {predicted_days}, Next Review Date: {next_review_date}")


        print(f"Predicted days: {predicted_days}, Next Review Date: {next_review_date}")

        supabase.table("user_topic_review_features").update({
            "next_review_date": next_review_date.isoformat(),
            "mastered": latest_score > 7
        }).eq("user_id", user_id).eq("topic_id", topic_id).execute()

        logging.info(f"✅ Model predicted next review date: {next_review_date}")
    else:
        logging.warning("⚠️ Could not fetch review features for model prediction.")


    if not status_update or not status_update.data:
        logging.warning(f"⚠️ Failed to update topic_status to '{new_status}' for topic {topic_id}")
        
     # Fetch the topic title from the topics table
    topic_lookup = supabase.table("topics").select("title").eq("topic_id", topic_id).maybe_single().execute()

    topic_title = topic_lookup.data["title"] if topic_lookup and topic_lookup.data else "your selected topic"
    mistake_count = total_questions - correct_count

    
    if email_id:
        subject = f"📚 Your Quiz Results — {topic_title}"

        html_body = f"""
        <div style="font-family: Arial, sans-serif; color: #333; padding: 20px; background-color: #f9f9f9;">
        <h2 style="color: #4CAF50;">🎉 Quiz Completed</h2>
        <h3 style="color: #222; margin-top: 0;">
            Topic: <span style="font-weight: bold; color: #1F4E79;">{topic_title}</span>
        </h3>

        <p>Hello,</p>
        <p>
            Thank you for completing the quiz on 
            <span style="font-weight: bold; color: #1F4E79;">{topic_title}</span>.
        </p>

        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #4CAF50; color: white;">
            <th style="text-align: left; padding: 8px;">Metric</th>
            <th style="text-align: left; padding: 8px;">Result</th>
            </tr>
            <tr style="background-color: #f2f2f2;">
            <td style="padding: 8px;">Score</td>
            <td style="padding: 8px;">
                <strong style="color: #2A4C9A;">{score:.2f} / 10</strong>
            </td>
            </tr>
            <tr>
            <td style="padding: 8px;">Mistaken Answers</td>
            <td style="padding: 8px;">{mistake_count} of {total_questions}</td>
            </tr>
            <tr style="background-color: #f2f2f2;">
            <td style="padding: 8px;">Mastery Status</td>
            <td style="padding: 8px;">{"✅ Mastered" if score > 7 else "🚧 Still Practicing"}</td>
            </tr>
            <tr>
            <td style="padding: 8px;">Next Review Date</td>
            <td style="padding: 8px;">
            <span style="color: #1A237E; font-weight: 500;">{next_review_date}</span>
            </td>
            </tr>
        </table>

        <p style="margin-top: 20px;">🚀 <strong>Keep it up!</strong> Every attempt brings you closer to your goal.</p>

        <blockquote style="border-left: 4px solid #4CAF50; margin: 20px 0; padding-left: 15px; color: #555;">
            "Success is the sum of small efforts, repeated day in and day out." — Robert Collier
        </blockquote>

        <p>Best regards,<br>Recallo , the Learning Platform Team</p>
        </div>
        """

        send_email(email_id, subject, html_body)

    return {
        "score": score,
        "total_questions": total_questions,
        "correct_answers": correct_count
    }
    
    

    