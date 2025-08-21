import os
import uuid
import logging
from flask import Flask, jsonify, request
from app import supabase, recent_file_uid
from upload_pdf import process_pdf

def process_uploaded_file(file, user_id, file_hash):
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    logging.info(f"📥 File saved: {file_path}")

    try:
        # Process your PDF file as usual
        success, chunk_count, uploaded_filename, file_uid = process_pdf(
            file_path, supabase, os.getenv("GEMINI_API_KEY"), user_id, file_hash
        )

        recent_file_uid = file_uid
        logging.info(f"File UUID stored: {recent_file_uid}")

        if success:
            logging.info(f"📌 recent_filename set to: {uploaded_filename}")
            logging.info(f"🗂️ File UUID: {file_uid}")
            return jsonify({"message": f"PDF processed. {chunk_count} chunks saved from '{uploaded_filename}'."}), 200
        else:
            return jsonify({"error": f"Failed to process PDF: {chunk_count}"}), 500

    except Exception as e:
        logging.error(f"Error during PDF processing: {str(e)}")
        return jsonify({"error": "Failed to process the PDF file."}), 500