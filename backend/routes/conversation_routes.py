from flask import Blueprint, request, jsonify, abort
from datetime import datetime
import logging
import uuid
import json
from utils.ai_helpers import generate_title
from utils.database_helpers import insert_chat_log_supabase_with_conversation
import config

conversation_bp = Blueprint('conversation_bp', __name__, url_prefix='/api/conversations')

from app import supabase, llm, memory, conversation
from utils.ai_helpers import get_summary, generate_questions, get_answer_from_file

@conversation_bp.route("", methods=["GET", "POST", "OPTIONS"])
def conversations():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response
    if request.method == "GET":
        user_id = request.args.get("user_id")
        if not user_id:
            abort(400, "Missing user_id")

        try:
            response = supabase.table("conversations").select(
                "conversation_id, title, created_at, updated_at"
            ).eq("user_id", user_id).order("updated_at", desc=True).execute()
            return jsonify(response.data), 200
        except Exception as e:
            logging.error(f"Error fetching conversations: {e}")
            abort(500, "Failed to fetch conversations")

    elif request.method == "POST":
        try:
            data = request.get_json()
            if not data:
                logging.error("No JSON data received in POST request")
                abort(400, "No JSON data provided")

            user_id = data.get("user_id")
            user_message = data.get("user_message", "")
            llm_response = data.get("llm_response", "")

            logging.info(f"Creating conversation for user_id: {user_id}")

            if not user_id:
                logging.error("Missing user_id in request data")
                abort(400, "Missing user_id")
        except Exception as e:
            logging.error(f"Error parsing request data: {e}")
            abort(400, "Invalid request data")

        title = generate_title(user_message, llm_response)

        new_conv_id = str(uuid.uuid4())

        try:
            response = supabase.table("conversations").insert({
                "conversation_id": new_conv_id,
                "user_id": user_id,
                "title": title
            }).execute()

            if response.data:
                return jsonify(response.data[0]), 201
            else:
                abort(500, "Failed to create conversation")
        except Exception as e:
            logging.error(f"Error creating conversation: {e}")
            abort(500, "Failed to create conversation")

@conversation_bp.route("/<conversation_id>", methods=["PUT", "DELETE"])
def update_or_delete_conversation(conversation_id):
    if request.method == "PUT":
        data = request.get_json()
        title = data.get("title")
        if not title:
            return jsonify({"error": "Title is required"}), 400

        try:
            supabase.table("conversations").update({
                "title": title,
                "updated_at": datetime.now().isoformat()
            }).eq("conversation_id", conversation_id).execute()
            return jsonify({"message": "Conversation renamed"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif request.method == "DELETE":
        try:
            supabase.table("chat_logs").delete().eq("conversation_id", conversation_id).execute()
            supabase.table("conversations").delete().eq("conversation_id", conversation_id).execute()
            return jsonify({"message": "Conversation deleted"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@conversation_bp.route("/<conv_id>/logs", methods=["GET"])
def get_chat_logs(conv_id):
    try:
        uuid.UUID(conv_id)
        logs = supabase.table("chat_logs").select(
            "user_message, response_message, created_at"
        ).eq("conversation_id", conv_id).order("created_at").execute()
        return jsonify(logs.data), 200
    except Exception as e:
        return jsonify({"error": "Invalid or missing conversation ID"}), 400