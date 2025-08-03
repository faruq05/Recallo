import re
import os
from dotenv import load_dotenv
import uuid
import json
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from datetime import datetime
import random

# Initialize Supabase client
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bhrwvazkvsebdxstdcow.supabase.co/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Generate UUID helper
def generate_uuid():
    return str(uuid.uuid4())

# Prompt template for 10 MCQ questions generation
prompt_template = PromptTemplate(
    input_variables=["content", "difficulty","number_of_questions"],
    template = """
        You are an intelligent quiz generator.

        Your task is to generate exactly {num_questions} multiple-choice questions (MCQs) from the following content.

        Instructions:
        - Questions should match the difficulty level: {difficulty}.
        - Questions should challenge the user's understanding, reasoning, or memory based on the content.
        - However, each question and its options must be:
        - Clearly written
        - Easy to read and comprehend
        - Focused on one idea at a time
        - Free from ambiguous or overly complex phrasing
        - Use formatting if needed for clarity:
            - Line breaks to separate context and query
            - Bullet points or steps
            - Code snippets or math formatting
        - Ensure that the **correct answer choices are randomly distributed** among A, B, C, and D.
        - Do not repeat the same correct answer option (like "B") for every question.
        - Vary the position of correct options to avoid predictability.

        Bad Example:
        ❌ "Which of the following is not unlikely to be dissimilar from the non-obvious behavior that contradicts the method pattern on Page 4?"

        Good Example:
        ✅ "Which of the following behaviors contradicts the expected behavior of the method described in the content?"

        Question Format:
        Question 1: [clear and understandable question]
        A) [option A]
        B) [option B]
        C) [option C]
        D) [option D]
        Answer: [Letter] - [Correct Answer Text]
        Explanation: Option B is correct because XYZ is defined as ABC in the text.

        Repeat this format for all 10 questions.

        Content:
        {content}
        """

    )


# Initialize LangChain LLM chain
def get_llm_chain(gemini_api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.3
    )
    return LLMChain(llm=llm, prompt=prompt_template)

# Parse LLM output into structured questions
def parse_mcq_response(text: str):
    questions = re.split(r"Question \d+:", text)[1:]
    parsed_questions = []

    for q in questions:
        q_text_match = re.search(r"^(.*?)(?:\nA\))", q, re.DOTALL)
        q_text = q_text_match.group(1).strip() if q_text_match else ""

        opts_match = re.findall(r"[A-D]\)\s*(.*)", q)
        options = opts_match if opts_match else []

        ans_match = re.search(r"Answer:\s*\[?([A-D])\]?\s*-\s*(.*)", q)
        correct_answer = ans_match.group(1) if ans_match else None
        correct_text = ans_match.group(2).strip() if ans_match else ""

        exp_match = re.search(r"Explanation:\s*(.*)", q)
        explanation = exp_match.group(1).strip() if exp_match else "No explanation provided."

        if q_text and options and correct_answer:
            parsed_questions.append({
                "question_text": q_text,
                "options": options,
                "correct_answer": correct_answer,
                "correct_text": correct_text,
                "explanation": explanation,
            })
    return parsed_questions

# Main function: generate & save questions for a topic
# def generate_and_save_mcqs(topic_id: str, gemini_api_key: str, difficulty_mode: str = "hard",user_id: str = None):
#     # Fetch topic content from Supabase
#     topic_res = supabase.table("topics").select("merged_content").eq("topic_id", topic_id).single().execute()
#     if not topic_res.data:
#         raise Exception(f"Topic with ID {topic_id} not found")

#     merged_content = topic_res.data["merged_content"]

#     # Initialize chain and generate questions
#     chain = get_llm_chain(gemini_api_key)
#     llm_response = chain.run(content=merged_content, difficulty=difficulty_mode)

#     # Parse questions
#     questions = parse_mcq_response(llm_response)
#     if len(questions) < 10:
#         raise Exception("Failed to generate 10 questions")

#     saved_questions = []

#     for q in questions:
#         qid = generate_uuid()
#         correct_index = ord(q["correct_answer"]) - ord('A')
#         correct_text = q["options"][correct_index] if 0 <= correct_index < len(q["options"]) else "N/A"

#         # Default explanation fallback if not provided
#         explanation = q.get("explanation") or f"The correct answer is option {q['correct_answer']} because it best reflects the core idea from the content."

#         supabase.table("quiz_questions").insert({
#             "question_id": qid,
#             "concept_id": topic_id,
#             "prompt": q["question_text"],
#             "answer": q["correct_answer"],
#             "answer_option_text": json.dumps({
#                 "A": q["options"][0],
#                 "B": q["options"][1],
#                 "C": q["options"][2],
#                 "D": q["options"][3]
#             }),
#             "explanation": explanation,
#             "created_at": datetime.now().isoformat()
#         }).execute()

#         saved_questions.append({
#             "question_id": qid,
#             "question_text": q["question_text"],
#             "options": q["options"],
#             "correct_answer": q["correct_answer"],
#             "answer_text": correct_text,  # ← fixed this
#             "explanation": explanation
#         })

#     return saved_questions


def generate_and_save_mcqs(topic_id: str, gemini_api_key: str, difficulty_mode: str = "hard", user_id: str = None):
    # Fetch topic content
    topic_res = supabase.table("topics").select("merged_content").eq("topic_id", topic_id).single().execute()
    if not topic_res.data:
        raise Exception(f"Topic with ID {topic_id} not found")

    merged_content = topic_res.data["merged_content"]
    mistake_questions = []

    # --- Get up to 4 wrong questions if attempted ---
    if user_id:
        attempt_res = supabase.table("quiz_attempts") \
            .select("attempt_id") \
            .eq("user_id", user_id) \
            .eq("topic_id", topic_id) \
            .order("submitted_at", desc=True) \
            .limit(1) \
            .execute()

        if attempt_res.data:
            last_attempt_id = attempt_res.data[0]["attempt_id"]
            wrong_ans_res = supabase.table("quiz_answers") \
                .select("question_id") \
                .eq("attempt_id", last_attempt_id) \
                .eq("is_correct", False) \
                .execute()

            if wrong_ans_res.data:
                wrong_qids = [a["question_id"] for a in wrong_ans_res.data]

                wrong_qs_res = supabase.table("quiz_questions") \
                    .select("question_id, prompt, answer, answer_option_text, explanation") \
                    .in_("question_id", wrong_qids) \
                    .execute()

                if wrong_qs_res.data:
                    mistake_questions = random.sample(wrong_qs_res.data, min(4, len(wrong_qs_res.data)))

        # --- Prepare LLM input ---
        mistake_count = len(mistake_questions)
        num_to_generate = 10 - mistake_count

        chain = get_llm_chain(gemini_api_key)
        llm_response = chain.run(content=merged_content, difficulty=difficulty_mode, num_questions=num_to_generate)

        # --- Parse generated questions ---
        questions = parse_mcq_response(llm_response)
        if len(questions) < num_to_generate:
            raise Exception(f"Expected {num_to_generate} questions, got {len(questions)}.")

        # --- Convert reused questions and prepend ---
        for reused in mistake_questions:
            options_map = json.loads(reused["answer_option_text"])
            parsed = {
                "question_text": reused["prompt"],
                "options": list(options_map.values()),
                "correct_answer": reused["answer"],
                "correct_text": options_map[reused["answer"]],
                "explanation": reused.get("explanation", "No explanation provided.")
            }
            questions.insert(0, parsed)

        # --- Save all 10 questions ---
        saved_questions = []

        for q in questions:
            qid = generate_uuid()
            correct_index = ord(q["correct_answer"]) - ord('A')
            correct_text = q["options"][correct_index] if 0 <= correct_index < len(q["options"]) else "N/A"

            explanation = q.get("explanation") or f"The correct answer is option {q['correct_answer']} because it best reflects the core idea from the content."

            supabase.table("quiz_questions").insert({
                "question_id": qid,
                "concept_id": topic_id,
                "prompt": q["question_text"],
                "answer": q["correct_answer"],
                "answer_option_text": json.dumps({
                    "A": q["options"][0],
                    "B": q["options"][1],
                    "C": q["options"][2],
                    "D": q["options"][3]
                }),
                "explanation": explanation,
                "created_at": datetime.now().isoformat()
            }).execute()

            saved_questions.append({
                "question_id": qid,
                "question_text": q["question_text"],
                "options": q["options"],
                "correct_answer": q["correct_answer"],
                "answer_text": correct_text,
                "explanation": explanation
            })

        return saved_questions