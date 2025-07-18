import re
import os
from dotenv import load_dotenv
import uuid
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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
    template=(
        "Based on the following content, generate exactly 10 multiple-choice questions.\n"
        "Each question must have 4 options labeled A, B, C, and D.\n"
        "Provide the correct answer letter for each question.\n"
        "Make the questions {difficulty}.\n"
        "Format output as:\n"
        "Question 1: ...\n"
        "A) ...\n"
        "B) ...\n"
        "C) ...\n"
        "D) ...\n"
        "Answer: [A-D]\n"
        "... repeat for all 10 questions\n\n"
        "Content:\n{content}\n"
    )
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

        ans_match = re.search(r"Answer:\s*\[?([A-D])\]?", q)
        correct_answer = ans_match.group(1) if ans_match else None

        if q_text and options and correct_answer:
            parsed_questions.append({
                "question_text": q_text,
                "options": options,
                "correct_answer": correct_answer,
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

    # Save questions to Supabase
    for q in questions:
        insert_res = supabase.table("quiz_questions").insert({
            "question_id": generate_uuid(),
            "concept_id": topic_id,  # Assuming topic_id maps to concept_id here
            "prompt": q["question_text"],
            "answer": q["correct_answer"],
            "created_at": None  # default now()
        }).execute()

        

    return questions
