from flask import Blueprint, request, jsonify, current_app
import logging

progress_bp = Blueprint('progress', __name__)

@progress_bp.route("/api/progress/<user_id>", methods=["GET"])
def get_user_progress(user_id):
    supabase = current_app.config['supabase']
    
    try:
        # Fetch all quiz attempts
        attempts_response = supabase.table("quiz_attempts") \
            .select("topic_id, score, submitted_at") \
            .eq("user_id", user_id) \
            .order("submitted_at", desc=True) \
            .execute()

        attempts = attempts_response.data or []

        if not attempts:
            return jsonify([]), 200

        topic_ids = list({a["topic_id"] for a in attempts})

        # Fetch topic metadata
        topics_response = supabase.table("topics") \
            .select("topic_id, title, file_name") \
            .in_("topic_id", topic_ids) \
            .execute()

        topics = topics_response.data or []

        return jsonify({
            "attempts": attempts,
            "topics": topics
        }), 200

    except Exception as e:
        logging.error(f"Error fetching progress: {e}")
        return jsonify({"error": "Failed to fetch progress"}), 500