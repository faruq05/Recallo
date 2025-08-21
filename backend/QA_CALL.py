# QA CALL
from QA_ANSWER import generate_and_save_mcqs
if __name__ == "__main__":
    import os

    TOPIC_ID = "3c2787be-56c2-45db-a577-552d1de5a6db"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # set your key here

    try:
        generated = generate_and_save_mcqs(TOPIC_ID, GEMINI_API_KEY, difficulty_mode="hard")
        print(f"Generated {len(generated)} questions:")
        for i, q in enumerate(generated, 1):
            print(f"Q{i}: {q['question_text']}")
            print(f"Options: {q['options']}")
            print(f"Answer: {q['correct_answer']}\n")
    except Exception as e:
        print("Error:", e)