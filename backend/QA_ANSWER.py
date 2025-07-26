import re
import os
from dotenv import load_dotenv
import uuid
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from datetime import datetime

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
    input_variables=["content", "difficulty"],
    template = """
        You are an intelligent quiz generator.

        Your task is to generate exactly 10 multiple-choice questions (MCQs) from the following content.

        Instructions:
        - Questions should match the difficulty level: {difficulty}.
        - Questions should challenge the user's understanding, reasoning, or memory based on the content.
        - However, each question and its options must be:
        - Clearly written
        - Easy to read and comprehend
        - Focused on one idea at a time
        - Free from ambiguous or overly complex phrasing
        - If needed, use formatting like:
        - Line breaks to separate question context and query
        - Bullet points or short numbered steps
        - Code snippets or math formatting for clarity
        - A one-line explanation of why that answer is correct, using: Explanation: [Your explanation]

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
def generate_and_save_mcqs(topic_id: str, gemini_api_key: str, difficulty_mode: str = "hard"):
    # Fetch topic content from Supabase
    topic_res = supabase.table("topics").select("merged_content").eq("topic_id", topic_id).single().execute()
    if not topic_res.data:
        raise Exception(f"Topic with ID {topic_id} not found")

    merged_content = topic_res.data["merged_content"]

    # Initialize chain and generate questions
    chain = get_llm_chain(gemini_api_key)
    llm_response = chain.run(content=merged_content, difficulty=difficulty_mode)

    # Parse questions
    questions = parse_mcq_response(llm_response)
    if len(questions) < 10:
        raise Exception("Failed to generate 10 questions")

    # Save questions to Supabase and collect full data to return
    saved_questions = []

    for q in questions:
        qid = generate_uuid()
        correct_index = ord(q["correct_answer"]) - ord('A')
        correct_text = q["options"][correct_index] if 0 <= correct_index < len(q["options"]) else "N/A"

        # Optional: Use a default explanation for now
        explanation_text = f"The correct answer is option {q['correct_answer']} because it best reflects the core idea from the content."

        supabase.table("quiz_questions").insert({
            "question_id": qid,
            "concept_id": topic_id,
            "prompt": q["question_text"],
            "answer": q["correct_answer"],
            "answer_option_text": q["correct_text"],
            "explanation": q["explanation"],
            "created_at": datetime.now().isoformat()
        }).execute()

        saved_questions.append({
            "question_id": qid,
            "question_text": q["question_text"],
            "options": q["options"],
            "correct_answer": q["correct_answer"],
            "answer_text": q["correct_text"],
            "explanation": q["explanation"]
        })

    return saved_questions