import uuid
import os
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from config import PINECONE_API_KEY  
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

import pinecone
print("pinecone module location:", pinecone.__file__)


# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client (using Pinecone class)
from pinecone import Pinecone, ServerlessSpec

# Create Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "document-index"

# Optionally, list the indexes and check if your index exists
if not pc.has_index(index_name):
    # If the index does not exist, create it
    pc.create_index(
        name='document-index',
        dimension=768,  # Adjust the dimension according to your embeddings
        metric='cosine',  # Choose the appropriate similarity metric (Euclidean, cosine, etc.)
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Choose your preferred cloud region
        )
    )

# Now you can access your Pinecone index
index = pc.Index('document-index')

def generate_unique_file_id():
    # Generate a unique UUID for each uploaded file
    return str(uuid.uuid4())  # Generate and return the UUID as a string

def process_pdf(file_path, supabase, gemini_api_key, user_id,file_hash):
    try:
        # Extract the filename automatically
        file_name = os.path.basename(file_path)

        # Step 1: Generate a unique identifier for the file
        file_uuid = generate_unique_file_id()

        # Step 2: Extract text from PDF
        reader = PdfReader(file_path)
        text = ""

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            
            # Check for empty or unreadable pages and log a warning
            if not page_text.strip():
                logging.warning(f"Page {page_num} in the PDF is empty or contains invalid text.")
            text += page_text
        
        # If no text was extracted, raise an error
        if not text.strip():
            raise ValueError("Empty or unreadable PDF.")

        # Step 3: Chunk text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        # Check if chunking failed (no chunks generated)
        if not chunks:
            raise ValueError("Chunking failed, no chunks generated.")

        # Step 4: Create LangChain Documents with metadata for body chunks and summary
        docs = [
            Document(page_content=chunk, metadata={"file_name": file_name, "tag": "body", "file_uuid": file_uuid, "user_id": user_id}) 
            for chunk in chunks
        ]
        
        # Debugging: Verify the metadata before storing
        logging.debug(f"Documents to be inserted: {[doc.metadata for doc in docs]}")

        embed_model=HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
       
        # Step 6: Generate embeddings for each document chunk
        embeddings = embed_model.embed_documents([doc.page_content for doc in docs])  # Generate embeddings for each chunk
        
        # Step 7: Prepare rows for Supabase insertion (with embeddings)
        rows = [
            {
                "chunk_id": file_uuid + f"_chunk_{i}",
                "content": doc.page_content,  # Document content
                "embedding": embedding,  # Embedding (vector for similarity search)
                "metadata": doc.metadata,  # Metadata (including file_uuid)
                "filename": file_name,  # Insert filename directly
                "file_uuid": file_uuid,  # Store the unique file UUID with each chunk
                "user_id": user_id,  # Store the user ID with each chunk
                "uploaded_at": datetime.now().isoformat(),
                "hash_file": file_hash
            }
            for i, (doc, embedding) in enumerate(zip(docs, embeddings))  # Loop over docs and embeddings
        ]
        
        # Step 8: Insert the rows into Supabase (store metadata)
        insert_response = supabase.table("documents").insert(rows).execute()

        logging.info(f"✅ Embedded and stored {len(docs)} chunks from '{file_name}' with unique UUID: {file_uuid}.")
        
        # Step 9: Upsert the embeddings into Pinecone (vector search)
        vectors = [
            (file_uuid + f"_chunk_{i}", embedding, {"user_id": user_id, "file_name": file_name, "file_uuid": file_uuid}) 
            for i, embedding in enumerate(embeddings)
        ]
        
        # Upsert vectors into Pinecone
        index.upsert(vectors=vectors)
        
        # Step 10: Clean up the uploaded PDF file (optional)
        os.remove(file_path)
        logging.info(f"🗑️ Deleted uploaded file: {file_path}")

        return True, len(docs), file_name, file_uuid  # Return the UUID along with other info

    except Exception as e:
        logging.error(f"🚫 PDF processing failed: {str(e)}")
        return False, str(e), None, None  # Return None for UUID on failure