from pinecone import Pinecone
from config import PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("document-index")

# Fetch all IDs
response = index.describe_index_stats()
total_vectors = response.get('total_vector_count', 0)

if total_vectors == 0:
    print("ℹ️ No vectors to delete.")
else:
    # Optional: fetch all vector IDs if your Pinecone setup supports metadata filtering
    print("⚠️ Deleting all vectors — this may take time...")
    index.delete(delete_all=True)
    print("✅ All vectors deleted.")
