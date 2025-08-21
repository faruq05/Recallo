from flask import Blueprint, request, jsonify, make_response
from datetime import datetime
import logging
import uuid
import re
import config

# Import the necessary helpers
from utils.database_helpers import insert_chat_log_supabase_with_conversation
from utils.ai_helpers import generate_title, get_summary, generate_questions_from_file, get_answer_from_file

chat_bp = Blueprint('chat_bp', __name__)

@chat_bp.route('/ask', methods=['POST'])
def ask():
    try:
        # Import supabase here to avoid circular dependency
        from app import supabase
        from app import llm

        user_message = request.json.get("message", "")
        user_id = request.json.get("user_id")
        conv_id = request.json.get("conversation_id")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        if not user_id:
            return jsonify({"error": "No user_id provided"}), 400

        # Validate or create conversation
        is_new_conversation = False
        if not conv_id:
            conv_id = str(uuid.uuid4())
            supabase.table("conversations").insert({
                "conversation_id": conv_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "title": "New Chat"
            }).execute()
            is_new_conversation = True
        else:
            existing_conv = supabase.table("conversations").select("conversation_id") \
                .eq("conversation_id", conv_id).eq("user_id", user_id).execute()
            if not existing_conv.data:
                conv_id = str(uuid.uuid4())
                supabase.table("conversations").insert({
                    "conversation_id": conv_id,
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "title": "New Chat"
                }).execute()
                is_new_conversation = True

        # List of phrases that indicate the user wants a summary or overview
        summary_keywords = [
            "summarize", "overview", "summary", "what is on this file", "what is this file about",
            "can you summarize", "give me a summary", "summarize the document", "overview of this file",
            "tell me about this file", "what's in this document", "can you give me the content of this file",
            "give me an overview", "give me the summary","summary of this file", "summarize this file", "summarize this document",
            "give me a summary of this file", "give me a summary of this document","summarise"
        ]

        # Expanded list of phrases indicating the user wants to generate questions
        question_keywords = [
            "generate questions","generate question", "make questions","make question", "create questions","create question", "ask questions",
            "can you make questions for this document", "can you generate questions from this",
            "create quiz questions", "generate quiz questions", "formulate questions from this",
            "please make some questions", "please generate questions", "can you suggest questions",
            "ask about this document", "can you create some questions", "create some questions for me",
            "formulate questions", "suggest questions based on this", "can you create a quiz",
            "can you ask some questions", "make some questions about this","create some questions","give me some questions",
            "generate questions from this file", "generate questions from this document", "make questions from this file",
            "make questions from this document", "create questions from this file", "create questions from this document",
            "ask questions about this file", "ask questions about this document", "can you generate questions from this file",
            "can you generate questions from this document", "can you make questions from this file", "can you make questions from this document",
        ]

        # Step 4: Route logic
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', user_message, re.IGNORECASE) for keyword in summary_keywords):
            summary_response, status_code = get_summary(user_id)
            response_text = summary_response.get_json()["response"]

        elif any(re.search(r'\b' + re.escape(keyword) + r'\b', user_message, re.IGNORECASE) for keyword in question_keywords):
            summary_response, status_code = generate_questions_from_file(user_id)
            response_text = summary_response.get_json()["response"]

        else:
            summary_response, status_code = get_answer_from_file(user_message, user_id)
            response_text = summary_response.get_json()["response"]

        # Step 5: Save to chat log
        insert_chat_log_supabase_with_conversation(supabase, conv_id, user_message, response_text)

        # Step 6: Generate title if this is first message
        if is_new_conversation:
            smart_title = generate_title(user_message, response_text)
            supabase.table("conversations").update({
                "title": smart_title
            }).eq("conversation_id", conv_id).execute()

        # Step 7: Return
        return jsonify({"response": response_text, "conversation_id": conv_id}), status_code

    except Exception as e:
        logging.error(f"❌ Ask route error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500