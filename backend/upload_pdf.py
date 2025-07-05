import os
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.schema import Document

def process_pdf(file_path, supabase, gemini_api_key):
    try:
        # Extract the filename automatically
        file_name = os.path.basename(file_path)

        # 1. Extract text from PDF
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])

        if not text.strip():
            raise ValueError("Empty or unreadable PDF.")

        # 2. Chunk text
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)


        # 3. Create LangChain Documents with metadata
        docs = [
            Document(page_content=chunk, metadata={"file_name": file_name})
            for chunk in chunks
        ]

        # 4. Embedding function
        embedding_fn = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )      

        # # 5. Store into Supabase with both metadata and filename column
        # vectorstore = SupabaseVectorStore(
        #     client=supabase,
        #     embedding=embedding_fn,
        #     table_name="documents",
        #     query_name="match_documents"
        # )

        # vectorstore.add_documents(
        #         docs,
        #         additional_columns={"filename": file_name})
       
        
        # logging.info(f"✅ Embedded and stored {len(docs)} chunks from '{file_name}'.")
        
        
        # 4. Embed the chunks manually
        texts = [doc.page_content for doc in docs]
        embeddings = embedding_fn.embed_documents(texts)
        
        # 5. Prepare rows with filename for Supabase
        rows = [
            {
                # optional if Supabase auto-generates IDs
                "content": doc.page_content,
                "embedding": embedding,
                "metadata": doc.metadata,
                "filename": file_name  # ✅ insert filename directly
            }
            for doc, embedding in zip(docs, embeddings)
        ]

        # 6. Insert into Supabase
        supabase.table("documents").insert(rows).execute()

        logging.info(f"✅ Embedded and stored {len(docs)} chunks from '{file_name}'.")

        # 6. Clean up
        os.remove(file_path)
        logging.info(f"🗑️ Deleted uploaded file: {file_path}")

        return True, len(docs)

    except Exception as e:
        logging.error(f"🚫 PDF processing failed: {str(e)}")
        return False, str(e)