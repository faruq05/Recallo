import logging
import uuid
from datetime import datetime

def insert_chat_log_supabase_with_conversation(supabase_client, user_id, conv_id, user_msg, resp_msg):
    try:
        message_id = str(uuid.uuid4())
        data = {
            "user_id": user_id,
            "conversation_id": conv_id,
            "user_message": user_msg,
            "response_message": resp_msg,
            "created_at": datetime.now().isoformat(),
            "message_id": message_id
        }
        supabase_client.table("chat_logs").insert(data).execute()

        supabase_client.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("conversation_id", conv_id).execute()

        logging.info(f"Inserted chat log into Supabase for conversation {conv_id}")
    except Exception as e:
        logging.error(f"Supabase insert error: {e}")