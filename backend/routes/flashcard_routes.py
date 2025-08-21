from flask import Blueprint, request, jsonify
import json
import logging
from datetime import datetime
import random
from langchain.schema import HumanMessage
from fetch_text_supabase import fetch_chunk_text_from_supabase
import config

flashcard_bp = Blueprint('flashcard_bp', __name__, url_prefix='/api')

from app import supabase, llm, embedding_fn

@flashcard_bp.route("/generate_flashcards", methods=["POST"])
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
        
        answers_response = supabase.from_("quiz_answers").select(
            "question_id, selected_answer, is_correct"
        ).eq("attempt_id", attempt_id).execute()

        quiz_answers = answers_response.data or []
        
        if not quiz_answers:
            return jsonify({"error": "No answers found for this attempt"}), 404
        
        question_ids = [qa["question_id"] for qa in quiz_answers]
        
        questions_response = supabase.from_("quiz_questions").select(
            "question_id, prompt, answer, explanation, answer_option_text, concept_id"
        ).in_("question_id", question_ids).execute()

        question_map = {q["question_id"]: q for q in questions_response.data}
        
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
            
        incorrect = [item for item in merged if item["is_correct"] is False]
        correct = [item for item in merged if item["is_correct"] is True]

        random.shuffle(incorrect)
        random.shuffle(correct)

        incorrect_sample = incorrect[:min(8, len(incorrect))]
        remaining = 10 - len(incorrect_sample)
        correct_sample = correct[:min(remaining, len(correct))]

        flashcards_base = incorrect_sample + correct_sample
        random.shuffle(flashcards_base)
        
        
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

        final_prompt = f"{system_prompt}\n\n📚 Topic Context:\n{merged_content or 'N/A'}\n\nQuiz Questions:\n{prompt_examples}"
        
        print("\n🧪 Sampled Flashcard Data:")
        for ex in examples:
            print(json.dumps(ex, indent=2))

        print("\n🧠 Final Prompt Sent to LLM:")
        print(final_prompt)

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

        return jsonify({"flashcards": flashcards, "message": "Generated"}), 200

    except Exception as e:
        logging.exception("Flashcard generation failed")
        return jsonify({"error": str(e)}), 500