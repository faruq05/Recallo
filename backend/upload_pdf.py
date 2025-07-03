# upload_pdf.py
import os
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.supabase import SupabaseVectorStore

def process_pdf(file_path, supabase, gemini_api_key):
    try:
        # 1. Extract text
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])

        if not text.strip():
            raise ValueError("Empty or unreadable PDF.")

        # 2. Chunk text
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([text])

        # 3. Embeddings
        embedding_fn = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )

        # 4. Save to Supabase
        vectorstore = SupabaseVectorStore.from_documents(
            docs,
            embedding=embedding_fn,
            client=supabase,
            table_name="documents",  # Ensure this exists in Supabase
            query_name="match_documents"  # Must exist as an RPC
        )

        logging.info(f"✅ Embedded and stored {len(docs)} chunks.")

        # 5. Delete the file
        os.remove(file_path)
        logging.info(f"🗑️ Deleted uploaded file: {file_path}")

        return True, len(docs)

    except Exception as e:
        logging.error(f"🚫 PDF processing failed: {str(e)}")
        return False, str(e)



