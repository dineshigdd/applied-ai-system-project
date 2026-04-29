import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_MASTER_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_MASTER_KEY missing from environment.")

    return create_client(url, key)
