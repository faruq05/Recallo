import logging
import os
import re
from flask import Flask, jsonify, request
from config import PINECONE_API_KEY
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage
from fetch_text_supabase import fetch_chunk_text_from_supabase
from app import llm, embedding_fn, supabase

def generate_title(user_message, llm_response):
    try:
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

def get_summary(user_id):
    try:
        print(f"Generating summary for user: {user_id}")
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
        return jsonify({"response": summary}), 200

    except Exception as e:
        logging.error(f"Error in get_summary: {e}")
        return jsonify({"error": "Failed to summarize"}), 500

def generate_questions_from_file(user_id):
    try:
        print(f"📌 Generating questions for user: {user_id}")
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

        docs_response = supabase.table('documents') \
            .select('content') \
            .eq('file_uuid', file_uuid) \
            .execute()

        docs = docs_response.data
        if not docs:
            return jsonify({"error": "No content found"}), 404

        concatenated_text = "\n\n".join(doc['content'] for doc in docs)
        
        prompt = f"""
        You are an expert educator with a deep understanding of how to generate relevant and insightful questions from a given text. Based on the following document, create a mixture of **Multiple Choice Questions (MCQs)** and **broad, open-ended questions** that focus on the most important topics discussed in the text. These questions should reflect the depth and complexity of the material, similar to what a professional-grade teacher would ask.

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
        return jsonify({"response": questions.strip()}), 200
    except Exception as e:
        logging.error(f"Error in generate_questions: {e}")
        return jsonify({"error": "Failed to generate questions"}), 500

def get_answer_from_file(user_query, user_id):
    try:
        user_id = user_id
        user_query = user_query
        print("get_answer_from_file called with user_query:", user_query, "and user_id:", user_id)

        if not user_query or not user_id:
            return jsonify({"error": "Missing user_query or user_id"}), 400

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("document-index")
        
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_fn
        )
        
        query_embedding = embedding_fn.embed_query(user_query)

        pinecone_results = index.query(
            vector=query_embedding,
            filter={"user_id": user_id},
            top_k=5,
            include_metadata=True
        )

        print(f"Pinecone Matches: {pinecone_results['matches']}")
        
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
            combined_text = "\n".join(relevant_docs)
            prompt = combined_text + "\n\nUser Question: " + user_query
            result = llm([HumanMessage(content=prompt)])

            return jsonify({
                "response": result.content,
                "source_documents": relevant_docs
            }), 200
        else:
            return jsonify({"response": "No relevant content found."}), 200

    except Exception as e:
        logging.error(f"Error in get_answer_from_file: {e}")
        return jsonify({"error": "Failed to fetch answer"}), 500