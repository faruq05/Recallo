from flask import Blueprint, request, jsonify
import hashlib
import os
import logging
from upload_pdf import process_pdf
import config

upload_bp = Blueprint('upload_bp', __name__)

# Assume these are passed in or accessible
from app import supabase, GEMINI_API_KEY, UPLOAD_FOLDER, ALLOWED_EXTENSIONS

# Helper functions for this module
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    global recent_file_uid
    logging.info("Upload route hit")
    user_id = request.form.get("user_id")
    print(f"User ID from upload: {user_id}")

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)

    from app import app
    if file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": "File is too large. Max size is 5MB"}), 413

    try:
        response = supabase.table('documents') \
            .select('id') \
            .eq('user_id', user_id) \
            .eq('hash_file', file_hash) \
            .execute()
        
        if response.data and len(response.data) > 0:
            logging.info(f"Duplicate upload detected for user {user_id} with file hash {file_hash}")
            return jsonify({"message": "You have already uploaded this file earlier."}), 409
    except Exception as e:
        logging.error(f"Error querying Supabase: {e}")
        return jsonify({"error": "Internal server error checking uploads."}), 500

    if file and allowed_file(file.filename):
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        logging.info(f"📥 File saved: {file_path}")

        try:
            success, chunk_count, uploaded_filename, file_uid = process_pdf(file_path, supabase, GEMINI_API_KEY, user_id, file_hash)
            
            # This global variable should be managed differently, perhaps within a user session or a database.
            # For now, let's keep it but note it's not ideal for a multi-user app.
            global recent_file_uid
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

    return jsonify({"error": "Invalid file type"}), 400