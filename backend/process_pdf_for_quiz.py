import os
import uuid
import logging
import re
from difflib import SequenceMatcher

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.cluster import AgglomerativeClustering
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def generate_unique_id():
    return str(uuid.uuid4())

def normalize_title(title):
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)  # remove punctuation
    title = " ".join(title.split())  # normalize whitespace
    return title

def is_similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() > threshold

def process_pdf_for_quiz(file_path, gemini_api_key, user_id, supabase, file_hash):
    try:
        file_name_full = os.path.basename(file_path)
        file_name = file_name_full.split("_", 1)[-1] if "_" in file_name_full else file_name_full
        file_uuid = generate_unique_id()

        # 1. Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        if not text.strip():
            raise ValueError("Empty or unreadable PDF.")

        # 2. Chunk splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        if not chunks:
            raise ValueError("No chunks generated.")

        # 3. Generate embeddings
        # embed_model = GoogleGenerativeAIEmbeddings(
        #     model="models/embedding-001",
        #     google_api_key=gemini_api_key
        # )
        embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embed_model.embed_documents(chunks)

        # 4. Clustering
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8)
        clustering.fit(embeddings)

        clustered_chunks = {}
        for idx, label in enumerate(clustering.labels_):
            clustered_chunks.setdefault(label, []).append(chunks[idx])
            
            
        MAX_RPM = 30
        total_requests = len(clustered_chunks)
        buffer = 2
        estimated_requests = total_requests + buffer

        if estimated_requests > MAX_RPM:
            return {
                "status": "warning",
                "message": (
                    f"Your PDF has {estimated_requests} topic groups to process, "
                    f"which exceeds the free tier limit of {MAX_RPM} requests per minute. "
                    "Please try a smaller file or upgrade to continue."
                )
            }

        # 5. Setup LLM & Prompts
        llm = ChatGroq(
            model="gemma2-9b-it", 
            api_key=groq_api_key,      
            temperature=0.3
        )

        # title_prompt = PromptTemplate(
        #     input_variables=["content"],
        #     template=(
        #         "From the following content, generate exactly ONE topic title.\n"
        #         "Constraints:\n"
        #         "- Maximum 5 words\n"
        #         "- No lists or explanations\n"
        #         "Content:\n{content}\n"
        #         "Topic Title:"
        #     )
        # )
        # summary_prompt = PromptTemplate(
        #     input_variables=["content"],
        #     template=(
        #         "Summarize the following content on the key points in 1-2 lines. "
        #         "Focus on what this topic is mainly about and what it covers.\n\n"
        #         "{content}\n\n"
        #         "Summary:"
        #     )
        # )
        
        # title_chain = LLMChain(llm=llm, prompt=title_prompt)
        # summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        
        combined_prompt = PromptTemplate(
            input_variables=["content"],
            template=(
                "Analyze the following content and perform two tasks:\n\n"
                "1. Generate exactly ONE topic title:\n"
                "- Maximum 5 words\n"
                "- No lists, explanations, or extra words\n\n"
                "2. Summarize the content in 1-2 lines:\n"
                "- Focus on what the topic is mainly about and what it covers\n\n"
                "Format your response strictly as:\n"
                "Title: <Your generated title>\n"
                "Summary: <Your summary>\n\n"
                "Content:\n{content}\n"
            )
        )
        
        combined_chain = LLMChain(llm=llm, prompt=combined_prompt)

        # 6. Generate and save topics
        rows = []
        existing_titles = []

        for cluster_label, chunk_list in clustered_chunks.items():
            merged_content = "\n".join(chunk_list)
            
            
            result= combined_chain.run(content=merged_content).strip()
            lines=result.split("\n")
            topic_title = next((line for line in lines if line.startswith("Title:")), "").replace("Title:", "").strip()
            topic_summary = next((line for line in lines if line.startswith("Summary:")), "").replace("Summary:", "").strip()

            # Generate title
            # topic_title = title_chain.run(content=merged_content).strip().replace("*", "").replace("-", "")
            if not topic_title:
                continue

            normalized_title = normalize_title(topic_title)
            if any(is_similar(normalized_title, normalize_title(t)) for t in existing_titles):
                logging.info(f"Skipping similar topic: {topic_title}")
                continue

            existing_titles.append(topic_title)

            # # Generate summary
            # topic_summary = summary_chain.run(content=merged_content).strip()

            # Add row
            rows.append({
                "topic_id": generate_unique_id(),
                "user_id": user_id,
                "document_for_quiz_id": file_uuid,
                "title": topic_title,
                "merged_content": merged_content,
                "topic_summary": topic_summary,
                "topic_status": "Ongoing",
                "file_name": file_name,
                "hash_file": file_hash
            })

            logging.info(f"Generated topic: {topic_title}")

        # 7. Insert into Supabase
        logging.info("Inserting topics into Supabase...")
        response = supabase.from_("topics").insert(rows).execute()
        logging.info(f"Insert response: {response}")

        # 8. Cleanup
        os.remove(file_path)
        return {"status": "success", "message": "Topics saved successfully."}

    except Exception as e:
        logging.error(f"PDF processing failed: {str(e)}")
        return {"status": "error", "message": str(e)}