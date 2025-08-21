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
        embed_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        embeddings = embed_model.embed_documents(chunks)

        # 4. Clustering
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6)
        clustering.fit(embeddings)

        clustered_chunks = {}
        for idx, label in enumerate(clustering.labels_):
            clustered_chunks.setdefault(label, []).append(chunks[idx])

        # 5. Setup LLM & Combined Prompt (ONE call returns Title + Summary)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.3
        )

        # keep your original wording/constraints; just combined into one prompt
        title_summary_prompt = PromptTemplate(
            input_variables=["content"],
            template=(
                "From the following content, generate exactly ONE topic title.\n"
                "Constraints:\n"
                "- Maximum 5 words\n"
                "- No lists or explanations\n"
                "Content:\n{content}\n"
                "Topic Title:\n\n"
                "Summarize the following content on the key points in 1-2 lines. "
                "Focus on what this topic is mainly about and what it covers.\n\n"
                "{content}\n\n"
                "Summary:"
            )
        )
        title_summary_chain = LLMChain(llm=llm, prompt=title_summary_prompt)

        # 6. Generate and save topics (ONE LLM call per cluster)
        rows = []
        existing_titles = []

        for cluster_label, chunk_list in clustered_chunks.items():
            merged_content = "\n".join(chunk_list)

            # One call for both title and summary
            resp = title_summary_chain.run(content=merged_content).strip()

            # Parse:
            # Expect output like:
            # Topic Title: <title line>
            # Summary: <one or more lines...>
            title_match = re.search(r"Topic Title:\s*(.+)", resp, re.IGNORECASE)
            summary_match = re.search(r"Summary:\s*([\s\S]+)$", resp, re.IGNORECASE)

            if not title_match or not summary_match:
                logging.info(f"Skipping malformed output: {resp[:200]}...")
                continue

            topic_title = title_match.group(1).strip().replace("*", "").replace("-", "")
            topic_summary = summary_match.group(1).strip()

            if not topic_title:
                continue

            normalized_title = normalize_title(topic_title)
            if any(is_similar(normalized_title, normalize_title(t)) for t in existing_titles):
                logging.info(f"Skipping similar topic: {topic_title}")
                continue

            existing_titles.append(topic_title)

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
