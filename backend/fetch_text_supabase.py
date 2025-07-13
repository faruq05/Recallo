def fetch_full_text_from_supabase(supabase, file_uuid, user_id):
    try:
        response = supabase.table("documents").select("content").eq("file_uuid", file_uuid).eq("user_id", user_id).execute()

        if response.data:
            print(response.data)
            return response.data[0]["content"]
        else:
            return None
    except Exception as e:
        print(f"Error fetching from Supabase: {e}")
        return None
