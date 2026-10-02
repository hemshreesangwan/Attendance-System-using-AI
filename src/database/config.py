import streamlit as st


supabase_url = st.secrets.get("SUPABASE_URL", "")
supabase_key = st.secrets.get("SUPABASE_KEY", "")

# Fallback to local offline mock database if credentials are missing or placeholder
if not supabase_url or "placeholder" in supabase_url:
    from src.database.mock_client import get_mock_client
    supabase = get_mock_client()
else:
    from supabase import create_client, Client
    supabase: Client = create_client(supabase_url, supabase_key)