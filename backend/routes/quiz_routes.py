from flask import Blueprint, request, jsonify, current_app, make_response
from flask_cors import cross_origin
import json
import logging
import random
from datetime import datetime, date
from QA_ANSWER import generate_and_save_mcqs
from matching_q_a import evaluate_and_save_quiz
from langchain.schema import HumanMessage

quiz_bp = Blueprint('quiz', __name__)

# --- Endpoint to update weak topics ---
@quiz_bp.route("/api/update-weak-topics", methods=["POST"])
def update_weak_topics():
    supabase = current_app.config['supabase']
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

# exam route
@quiz_bp.route("/generate-questions", methods=['POST', 'OPTIONS'])
def generate_questions():
    GEMINI_API_KEY = current_app.config['GEMINI_API_KEY']
    
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
@quiz_bp.route("/submit-answers", methods=["POST", "OPTIONS"])
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
@quiz_bp.route("/api/answer-analysis")
def get_answer_analysis():
    supabase = current_app.config['supabase']
    
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
@quiz_bp.route("/api/generate_flashcards", methods=["POST"])
def generate_flashcards():
    supabase = current_app.config['supabase']
    # llm = current_app.config['llm']
    llm= current_app.config['llm1']
    
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
        system_prompt =""" You are a study assistant that generates concept flashcards from quiz questions.

            Your job:
            - Analyze the provided quiz questions and answers.
            - Identify the **10 most important core concepts** from the questions.
            - For each concept, create a structured flashcard with clear explanations.

            Each flashcard must strictly follow this JSON structure:
            [
            {
                "core_concept": "The fundamental idea or topic being tested, stated clearly and concisely.",
                "key_theory": "A clear, accurate explanation of the concept, written for learning and retention.",
                "common_mistake": "The most common misunderstanding or error related to this concept. Only include this field if the user answered incorrectly. Omit entirely if not relevant."
            },
            ...
            ]

            ⚙️ **Rules & Requirements:**
            1. **Exactly 10 items** — no more, no less.
            2. Each item must be **unique** — do NOT repeat concepts.
            3. Output must be **pure JSON only**:
            - No markdown.
            - No extra commentary.
            - The very first character must be `[` and the last character must be `]`.
            4. If a concept was answered correctly, **omit** `"common_mistake"` entirely for that flashcard.
            5. All explanations must be **clear, concise, and educational**, as if teaching a beginner.
            6. Stop generating immediately after the 10th flashcard.

            💡 **Summary:**
            Your response should always be a clean JSON array of 10 educational flashcards, each with:
            - `core_concept` — what to learn
            - `key_theory` — why it matters
            - `common_mistake` — only if relevant"""


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