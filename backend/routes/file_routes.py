from flask import Blueprint, request, jsonify, current_app
import os
import uuid
import hashlib
import logging
from upload_pdf import process_pdf
from process_pdf_for_quiz import process_pdf_for_quiz

file_bp = Blueprint('file', __name__)

@file_bp.route('/upload', methods=['POST'])
def upload_file():
    supabase = current_app.config['supabase']
    GEMINI_API_KEY = current_app.config['GEMINI_API_KEY']
    allowed_file = current_app.config['allowed_file']
    
    current_app.config['recent_file_uid'] = None  # Reset global variable
    
    logging.info("Upload route hit")
    user_id = request.form.get("user_id")  # Get user ID from the form data
    print(f"User ID from upload: {user_id}")

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read file bytes to compute hash
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)  # Reset file pointer after reading

    # Check file size (adjust if needed since content_length might not always be available)
    if file.content_length and file.content_length > current_app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": "File is too large. Max size is 5MB"}), 413

    # Check if file hash already exists for this user in Supabase
    try:
        response = supabase.table('documents') \
            .select('id') \
            .eq('user_id', user_id) \
            .eq('hash_file', file_hash) \
            .execute()
        
        if response.data and len(response.data) > 0:
            # File already uploaded by this user
            logging.info(f"Duplicate upload detected for user {user_id} with file hash {file_hash}")
            return jsonify({"message": "You have already uploaded this file earlier."}), 409
    except Exception as e:
        logging.error(f"Error querying Supabase: {e}")
        return jsonify({"error": "Internal server error checking uploads."}), 500

    # Validate file extension/type
    if file and allowed_file(file.filename):
        upload_folder = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        logging.info(f"📥 File saved: {file_path}")

        try:
            # Process your PDF file as usual
            success, chunk_count, uploaded_filename, file_uid = process_pdf(file_path, supabase, GEMINI_API_KEY, user_id,file_hash)

            current_app.config['recent_file_uid'] = file_uid  # Store in app config
            logging.info(f"File UUID stored: {file_uid}")

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

# topic jsx route
@file_bp.route('/quiz-question', methods=['POST'])
def quiz_question():
    supabase = current_app.config['supabase']
    GEMINI_API_KEY = current_app.config['GEMINI_API_KEY']
    allowed_file = current_app.config['allowed_file']
    
    user_id = request.form.get("user_id")
    file = request.files.get("file")

    if not user_id or not file:
        return jsonify({"error": "Missing user_id or file."}), 400
    
    # Read file bytes to compute hash
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)  # Reset file pointer after reading

    if len(file_bytes) > current_app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": "File is too large. Max size is 5MB."}), 413
    
    
    # Check if file hash already exists for this user in Supabase
    try:
        response = supabase.table('topics') \
            .select('topic_id') \
            .eq('user_id', user_id) \
            .eq('hash_file', file_hash) \
            .execute()
        
        if response.data and len(response.data) > 0:
            # File already uploaded by this user
            logging.info(f"Duplicate upload detected for user {user_id} with file hash {file_hash}")
            return jsonify({"message": "You have already uploaded this file earlier."}), 409
    except Exception as e:
        logging.error(f"Error querying Supabase: {e}")
        return jsonify({"error": "Internal server error checking uploads."}), 500

    if  allowed_file(file.filename):
        # Save file temporarily
        if not os.path.exists(current_app.config['UPLOAD_FOLDER']):
            os.makedirs(current_app.config['UPLOAD_FOLDER'])

        temp_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)

        # Process PDF for quiz topics and save them to Supabase
        result = process_pdf_for_quiz(temp_path, GEMINI_API_KEY, user_id, supabase,file_hash)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logging.info(f"🗑️ Deleted temporary file: {temp_path}")

        if result:
            return jsonify(result),200
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to process PDF."
            }), 500

    return jsonify({"error": "Invalid file type."}), 400