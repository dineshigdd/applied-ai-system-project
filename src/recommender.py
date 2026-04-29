import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import google.generativeai as genai
import os

@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_tempo: Optional[float] = None

def load_songs(csv_path: str) -> List[Dict]:
    """
    Keeping this for backward compatibility, though in RAG 
    your data usually lives in the cloud (Supabase).
    """
    import csv
    songs = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                songs.append({
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row.get("valence", 0)),
                    "danceability": float(row.get("danceability", 0)),
                    "acousticness": float(row.get("acousticness", 0)),
                })
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found.")
    return songs

def compute_active_weight(user_prefs: Dict) -> float:
    """
    In RAG, the 'weight' is usually 1.0 (the similarity score).
    We return 1.0 here so your main.py math (score/active_weight) * 10 works.
    """
    return 1.0

def recommend_songs(user_prefs: dict, songs: list, k: int = 5, supabase=None, model=None) -> list:
    """
    The RAG Engine.
    1. Converts user preferences into a natural language string.
    2. Embeds that string into a vector.
    3. Queries Supabase using the match_music RPC.
    """
    if not supabase or not model:
        raise ValueError("RAG requires supabase_client and embedding_model.")

    # 1. Construct the Search Query
    # We turn the structured dict into a sentence the AI can 'understand'
    query_text = (
        f"A {user_prefs.get('genre')} song with a {user_prefs.get('mood')} mood. "
        f"Energy level is {user_prefs.get('energy')}. "
    )
    if user_prefs.get('target_tempo'):
        query_text += f"Tempo is around {user_prefs.get('target_tempo')} BPM."

    # 2. Vectorize the User Intent
    query_vector = model.encode(query_text).tolist()

    # 3. Call the Supabase Vector Search (RPC)
    # This replaces your old 'score_song' loop
    response = supabase.rpc("match_music", {
        "query_embedding": query_vector,
        "match_threshold": 0.4, # Broaden search to ensure we get results
        "match_count": k
    }).execute()

    # 4. Format for your main.py Interface
    # Your interface expects: (song_dict, score, explanation)
    formatted_results = []
    for item in response.data:
        # We use the similarity score as the ranking metric
        score = item['similarity'] 
        
        # NEW: Call Gemini for a real explanation!
        explanation = generate_explanation(user_prefs, item)
        
        formatted_results.append((item, score, explanation))

    return formatted_results

def generate_explanation(user_prefs: dict, song: dict) -> str:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')

    # We use .get() so it returns 'N/A' instead of crashing if a key is missing
    prompt = f"""
    You are a music expert. A user wants: 
    Genre: {user_prefs.get('genre', 'Any')}, Mood: {user_prefs.get('mood', 'Any')}.
    (Secondary Energy Target: {user_prefs.get('energy', 'Balanced')})

    I found this song:
    Title: {song.get('title', 'Unknown')} by {song.get('artist', 'Unknown')}
    Actual Genre: {song.get('genre', 'Unknown')}, Actual Mood: {song.get('mood', 'Unknown')}
    
    In one short sentence, explain why this song is a perfect match for the user's requested genre and mood.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return f"This {song.get('genre')} track perfectly captures the {user_prefs.get('mood')} vibe you're looking for."
    
class Recommender:
    """
    The Class-based wrapper used by your test suite or advanced logic.
    """
    def __init__(self, supabase_client, embedding_model):
        self.supabase = supabase_client
        self.model = embedding_model

    def recommend(self, user: UserProfile, k: int = 5) -> List[Dict]:
        # Convert dataclass to dict for the recommend_songs function
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "target_tempo": user.target_tempo
        }
        return recommend_songs(prefs, [], k, self.supabase, self.model)