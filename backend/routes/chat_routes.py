from flask import Blueprint, request, jsonify, current_app
import uuid
import logging
import re
import os
# import time
from datetime import datetime
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage
from fetch_text_supabase import fetch_chunk_text_from_supabase
from config import PINECONE_API_KEY

chat_bp = Blueprint('chat', __name__)

# --- Supabase Insert Helper ---
def insert_chat_log_supabase_with_conversation(user_id, conv_id, user_msg, resp_msg):
    """Insert chat log with conversation tracking"""
    try:
        supabase = current_app.config['supabase']
        message_id = str(uuid.uuid4())
        data = {
            "user_id": user_id,
            "conversation_id": conv_id,
            "user_message": user_msg,
            "response_message": resp_msg,
            "created_at": datetime.now().isoformat(),
            "message_id": message_id
        }
        supabase.table("chat_logs").insert(data).execute()

        # Update conversation timestamp
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("conversation_id", conv_id).execute()

        logging.info(f"Inserted chat log into Supabase for conversation {conv_id}")
    except Exception as e:
        logging.error(f"Supabase insert error: {e}")

def generate_title(user_message, llm_response):
    try:
        llm = current_app.config['llm']
        prompt = f"""
        Generate a short and meaningful title (3 to 6 words max) for a conversation based on this exchange:

        User: {user_message}
        Assistant: {llm_response}

        The title should be concise, informative, and not include any quotation marks.
        """
        result = llm.predict(prompt)
        return result.strip().replace('"', '')
    except Exception as e:
        logging.warning(f"⚠️ Failed to generate title: {e}")
        return "New Chat"

@chat_bp.route('/chat', methods=['POST','OPTIONS'])
def chat():
    
    if request.method == "OPTIONS":
            response = jsonify({})
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
            return response   
    try:
        supabase = current_app.config['supabase']
        conversation = current_app.config['conversation']
        
        if not request.json:
            logging.error("No JSON data received in chat request")
            return jsonify({"error": "No JSON data provided"}), 400
        user_message = request.json.get("message", "")
        user_id = request.json.get("user_id")
        conv_id = request.json.get("conversation_id")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        # if not user_id:
        #     return jsonify({"error": "No user ID provided"}), 400
        
        logging.info(f"Received message: {user_message}")
        
        # Step 1: Handle conversation ID safely
        if not conv_id:
            conv_id = str(uuid.uuid4())
            if user_id:
                supabase.table("conversations").insert({
                    "conversation_id": conv_id,
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "title": "New Chat"
                }).execute()

        # Enhanced educational and conversational prompt
        prompt = f"""
        You are an AI assistant named Recallo. Always refer to yourself as Recallo. With a deep knowledge base designed to help users learn and understand any topic in the most effective and engaging way. Your role is to provide clear, accurate, and detailed explanations, making complex topics easy to understand. 

        Respond to the user with a friendly, conversational tone, as if you're explaining the concept to a student. Break down the topic step by step when necessary, and give real-life examples to aid comprehension. Also, offer YouTube video suggestions that are relevant to the topic for further learning.

        Be empathetic, patient, and provide well-rounded answers. If the user asks for clarifications or examples, be ready to offer more detailed responses and give helpful suggestions.

        User message: {user_message}
        """
        
        
        # Run the conversation chain with memory
        # start_time = time.time()
        ai_reply = conversation.predict(input=user_message)
        # end_time = time.time()
        # elapsed = round(end_time - start_time, 2)  # seconds, rounded
        # logging.info(f"AI reply generated in {elapsed} seconds")
        # Step 3: Insert log
        if user_id:
            insert_chat_log_supabase_with_conversation(user_id, conv_id, user_message, ai_reply)

        # Step 4: If this is the first message, generate smart title
            log_check = supabase.table("chat_logs").select("id").eq("conversation_id", conv_id).execute()
            if log_check.data and len(log_check.data) == 1:
                smart_title = generate_title(user_message, ai_reply)
                supabase.table("conversations").update({
                    "title": smart_title
                }).eq("conversation_id", conv_id).execute()


        return jsonify({
            "response": ai_reply,
            "conversation_id": conv_id
        }), 200


    except Exception as e:
        logging.error(f"/chat error: {e}")
        return jsonify({"error": "Something went wrong"}), 500

# chat mode route
@chat_bp.route('/ask', methods=['POST'])
def ask():
    try:
        supabase = current_app.config['supabase']
        
        user_message = request.json.get("message", "")
        user_id = request.json.get("user_id")
        conv_id = request.json.get("conversation_id")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        if not user_id:
            return jsonify({"error": "No user_id provided"}), 400

        # Validate or create conversation
        if conv_id:
            existing_conv = supabase.table("conversations").select("conversation_id") \
                .eq("conversation_id", conv_id).eq("user_id", user_id).execute()
            if not existing_conv.data:
                conv_id = None

        # Step 2: Create new conversation with placeholder title
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
            summary_response, status_code = generate_questions(user_id)
            response_text = summary_response.get_json()["response"]

        else:
            summary_response, status_code = get_answer_from_file(user_message, user_id)
            response_text = summary_response.get_json()["response"]

        # Step 5: Save to chat log
        insert_chat_log_supabase_with_conversation(user_id, conv_id, user_message, response_text)

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

# === Functional Chains ===
def get_summary(user_id):
    try:
        supabase = current_app.config['supabase']
        llm = current_app.config['llm']
        
        print(f"Generating summary for user: {user_id}")

        # Step 1: Get the most recently uploaded file for this user
        response = supabase.table('documents') \
            .select('file_uuid') \
            .eq('user_id', user_id) \
            .order('uploaded_at', desc=True) \
            .limit(1) \
            .execute()

        if not response.data:
            return jsonify({"error": "No recent file uploaded."}), 400

        file_uuid = response.data[0]['file_uuid']
        print("📄 Using file UUID:", file_uuid)

        # Step 2: Get all chunks for that file
        chunks_res = supabase.table('documents') \
            .select('content') \
            .eq('file_uuid', file_uuid) \
            .execute()

        docs = chunks_res.data
        if not docs:
            return jsonify({"error": "No content found for this file."}), 404

        concatenated_text = "\n\n".join(doc['content'] for doc in docs)
        prompt = f"""
        You are an expert assistant with a deep understanding of how to summarize complex documents in a clear, concise, and professional manner. Based on the following content, summarize the key points in a way that is easy to understand, highlighting the most important information while keeping the summary brief and to the point.

        Please focus on:
        1. Extracting the core ideas and themes.
        2. Presenting the summary in a structured format with key takeaways.
        3. Avoiding unnecessary details or long explanations.

        Make sure the summary is **short and professional**, providing only the most relevant and actionable insights.

        Here is the content you need to summarize:

        ======== PDF Content ========
        {concatenated_text}
        =============================

        Your summary:
        """

        summary = llm.predict(prompt)
        summary = summary.strip()


        # Optional: log the interaction
        # insert_chat_log("Summarize Request", summary)

        return jsonify({"response": summary}), 200

    except Exception as e:
        logging.error(f"Error in get_summary: {e}")
        return jsonify({"error": "Failed to summarize"}), 500

def generate_questions(user_id):
    try:
        supabase = current_app.config['supabase']
        llm = current_app.config['llm']
        
        print(f"📌 Generating questions for user: {user_id}")

        # Step 1: Get the most recently uploaded file for this user
        response = supabase.table('documents') \
            .select('file_uuid') \
            .eq('user_id', user_id) \
            .order('uploaded_at', desc=True) \
            .limit(1) \
            .execute()

        if not response.data:
            return jsonify({"error": "No recent file uploaded."}), 400

        file_uuid = response.data[0]['file_uuid']
        print(f"✅ Using file UUID: {file_uuid}")

        # Step 2: Get all chunks for that file
        docs_response = supabase.table('documents') \
            .select('content') \
            .eq('file_uuid', file_uuid) \
            .execute()

        docs = docs_response.data
        if not docs:
            return jsonify({"error": "No content found"}), 404

        concatenated_text = "\n\n".join(doc['content'] for doc in docs)
        
        prompt = f"""
        You are an expert at generating questions from a given text. Based on the following document, create relevant and insightful questions that can be asked:You are an expert educator with a deep understanding of how to generate relevant and insightful questions from a given text. Based on the following document, create a mixture of **Multiple Choice Questions (MCQs)** and **broad, open-ended questions** that focus on the most important topics discussed in the text. These questions should reflect the depth and complexity of the material, similar to what a professional-grade teacher would ask.

        For each question:

        1. Provide the **correct answer** at the end of the question on a **separate line**.
        2. **Explain why** the answer is correct for **broad questions**, with a focus on the key concepts behind the question. Keep the explanation concise, just a few sentences.
        3. Specify **which topic** the question is related to on a **separate line with a line gap** (e.g., "Topic: [Topic Name]").
        4. Ensure the questions are well-structured, engaging, and relevant to the content.
        5. If applicable, provide at least **one resource or website** where users can learn more about each topic (this is **optional** and should be on a **separate new line with a line gap** if included).
        6. **For MCQs**, do not provide explanations—just the question and the answer on separate lines.
        7. **For broad questions**, make sure the explanation is concise, focusing on key ideas.
        8. Options will be on a separate line and make gap between the 4 options as well as the question so that it is easy to read.
        9. It is not mandatory to generate 10 questions, but try to generate at least 5 questions.

        Here is the content you need to generate questions from:

        ======== PDF Content ========
        {concatenated_text}
        =============================

        Generated Questions and Answers will be in the following format:

        1. **Question**: [Insert Question Here]

        **Answer**: [Insert Correct Answer Here]

        **Topic**: [Insert Topic Name Here]

        [Optional: **Learn More**: [Insert Learning Resource or Website Here]]

        2. **Question**: [Insert Question Here]

        **Answer**: [Insert Correct Answer Here]

        **Explanation**: [Provide Explanation for the Answer in a Few Sentences]

        **Topic**: [Insert Topic Name Here]

        [Optional: **Learn More**: [Insert Learning Resource or Website Here]]

        [Repeat as needed for additional questions]
        """
        questions = llm.predict(prompt)
        # insert_chat_log("Generate Questions Request", questions.strip())
        return jsonify({"response": questions.strip()}), 200
    except Exception as e:
        logging.error(f"Error in generate_questions: {e}")
        return jsonify({"error": "Failed to generate questions"}), 500

def get_answer_from_file(user_query, user_id):
    try:
        supabase = current_app.config['supabase']
        embedding_fn = current_app.config['embedding_fn']
        llm1 = current_app.config['llm1']
        
        # # Get input from the request body
        # user_query = request.json.get("user_query")
        # user_id = request.json.get("user_id")
        userID=user_id;
        userQuery= user_query;
        print("get_answer_from_file called with user_query:", user_query, "and user_id:", user_id)

        if not user_query or not user_id:
            return jsonify({"error": "Missing user_query or user_id"}), 400


        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Get existing index
        index = pc.Index("document-index")

        # Set up Langchain Pinecone VectorStore
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_fn
        )

        query_embedding = embedding_fn.embed_query(userQuery)

        pinecone_results = index.query(
            vector=query_embedding,
            filter={"user_id": userID},
            top_k=5,
            include_metadata=True
        )

        print(f"Pinecone Matches: {pinecone_results['matches']}")
        # Check if no matches found
        if not pinecone_results['matches']:
            return jsonify({"response": "No relevant content found."}), 200


        relevant_docs = []
        for match in pinecone_results['matches']:
            chunk_id = match['id']
            chunk_data = fetch_chunk_text_from_supabase(supabase, chunk_id, user_id)

            if chunk_data:
                relevant_docs.append(chunk_data)
                print(f"Chunk Text: {chunk_data}")
            

            if relevant_docs:
                # Concatenate or process relevant_docs as needed before LLM
                combined_text = "\n".join(relevant_docs)
                # Build chat message properly
                prompt = combined_text + "\n\nUser Question: " + user_query
                result = llm1([HumanMessage(content=prompt)])

            return jsonify({
                "response": result.content,
                "source_documents": relevant_docs
            }), 200
        else:
            return jsonify({"response": "No relevant content found."}), 200

    except Exception as e:
        logging.error(f"Error in get_answer_from_file: {e}")
        return jsonify({"error": "Failed to fetch answer"}), 500