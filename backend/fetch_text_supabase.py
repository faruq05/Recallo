def fetch_chunk_text_from_supabase(supabase, chunk_id, user_id):
    try:
        response = supabase.table("documents") \
            .select("content") \
            .eq("chunk_id", chunk_id) \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()

        if response.data:
            print(response.data)
            return response.data[0]["content"]
        else:
            return None
    except Exception as e:
        print(f"Error fetching from Supabase: {e}")
        return None
