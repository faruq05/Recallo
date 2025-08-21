from matching_q_a import evaluate_and_save_quiz
if __name__ == "__main__":
    user_id = "b9ddffb4-f3b8-45a3-aad7-d9153f786c71"  # your user id
    topic_id = "fe0f16a0-735f-4154-ba13-cd9a91cdf798"  # your topic id
    
    submitted_answers_test = [
    {"question_id": "fdcf427c-29b4-4bd3-90c8-81a1f3f2b489", "selected_answer": "A"},
    {"question_id": "a1004890-8fb6-4405-85f7-0de6850b6208", "selected_answer": "C"},
    {"question_id": "28bcd5fa-6ea6-4c9c-ac44-5f2ed241f452", "selected_answer": "A"},
    {"question_id": "2b157266-cbdb-45eb-8644-9413be207ec5", "selected_answer": "B"},
    {"question_id": "493550c6-fd42-45f7-805d-348e269922a5", "selected_answer": "C"},
    {"question_id": "95c1b68a-45c5-473d-becc-4b57bcd39d12", "selected_answer": "B"},
    {"question_id": "1411b67c-a856-4bfe-a8ce-a94d9b73793b", "selected_answer": "C"},
    {"question_id": "9c6b11e4-2b84-4054-b75e-4c2e5d18a938", "selected_answer": "C"},
    {"question_id": "86e0325b-84f5-4015-98f1-3080f329e89c", "selected_answer": "D"},
    {"question_id": "dd41e724-ebb6-4432-ad98-58655a9e09aa", "selected_answer": "C"},
]


    try:
        result = evaluate_and_save_quiz(user_id, topic_id, submitted_answers_test)
        print("Test Result:", result)
    except Exception as e:
        print("Test Failed:", e)
