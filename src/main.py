import os
from supabase import create_client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
# Import from your recommender.py
from .recommender import load_songs, recommend_songs, compute_active_weight

# --- THE PROFILES LIST MUST BE HERE ---
PROFILES = [
    {
        "name": "Gym Session",
        "prefs": {
            "genre": "pop",
            "mood": "intense",
            "energy": 0.92,
            "likes_acoustic": False,
            "target_tempo": 135,
        },
    },
    {
        "name": "Late Night Study",
        "prefs": {
            "genre": "lofi",
            "mood": "focused",
            "energy": 0.38,
            "likes_acoustic": True,
            "target_tempo": 78,
        },
    },
    {
        "name": "Sunday Morning",
        "prefs": {
            "genre": "bossa nova",
            "mood": "dreamy",
            "energy": 0.30,
            "likes_acoustic": True,
            "target_tempo": 75,
        },
    },
    {
        "name": "Road Trip",
        "prefs": {
            "genre": "rock",
            "mood": "energetic",
            "energy": 0.88,
            "likes_acoustic": False,
            "target_tempo": 150,
        },
    },
    {
        "name": "[EDGE] High Energy + Sad Mood",
        "prefs": {
            "genre": "blues",
            "mood": "sad",
            "energy": 0.90,
            "likes_acoustic": False,
            "target_tempo": 155,
        },
    },
    {
        "name": "[EDGE] Unknown Genre",
        "prefs": {
            "genre": "k-pop",
            "mood": "happy",
            "energy": 0.75,
            "likes_acoustic": False,
            "target_tempo": 120,
        },
    },
    {
        "name": "[EDGE] Ambiguous Energy (0.5)",
        "prefs": {
            "genre": "jazz",
            "mood": "relaxed",
            "energy": 0.50,
            "likes_acoustic": None,
            "target_tempo": None,
        },
    },
    {
        "name": "[EDGE] Acoustic + High Energy",
        "prefs": {
            "genre": "folk",
            "mood": "uplifting",
            "energy": 0.92,
            "likes_acoustic": True,
            "target_tempo": 140,
        },
    },
]

def print_recommendations(profile_name: str, user_prefs: dict, songs: list, k: int = 3, supabase=None, model=None) -> None:
    # Use the RAG-enabled recommend_songs
    recommendations = recommend_songs(user_prefs, songs, k=k, supabase=supabase, model=model)

    print()
    print("=" * 60)
    print(f" 🚀 PROFILE : {profile_name.upper()}")
    # We keep the user's requested energy here so the intent is clear
    print(f"    Target: {user_prefs.get('genre')} | {user_prefs.get('mood')} | Energy: {user_prefs.get('energy')}")
    print("=" * 60)

    # In RAG mode, compute_active_weight returns 1.0
    active_weight = compute_active_weight(user_prefs)
    
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        # Use .get() for title and artist to prevent KeyErrors
        title = song.get('title', 'Unknown Title')
        artist = song.get('artist', 'Unknown Artist')
        
        print(f"\n  #{rank}  {title} — {artist}")
        
        # similarity (0-1) * 10 gives us a nice 0-10 score
        normalized = (score / active_weight) * 10
        print(f"      Score : {normalized:.1f} / 10")
        
        # CLEANED UI: We only show Genre and Mood from the DB
        db_genre = song.get('genre', 'N/A')
        db_mood = song.get('mood', 'N/A')
        print(f"      Tags  : {db_genre} | {db_mood} ")
        
        print(f"      Why   :")
        # Gemini usually returns a clean string; we split by ";" just in case
        for reason in explanation.split(";"):
            if reason.strip():
                print(f"        • {reason.strip()}")

    print("\n" + "=" * 60)


def main() -> None:
    # Load Environment variables
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_MASTER_KEY")
    
    # Check if keys exist
    if not url or not key:
        print("Error: SUPABASE_URL or SUPABASE_MASTER_KEY missing from .env file.")
        return

    # 1. Setup Supabase and Model
    print("Connecting to Supabase...")
    supabase = create_client(url, key)
    
    print("Loading AI Model (Jina v5)... this may take a moment...")
    model = SentenceTransformer('jinaai/jina-embeddings-v5-text-small-retrieval', trust_remote_code=True)
    
    # 2. Load songs (still useful for local reference if needed)
    songs = load_songs("data/songs.csv")

    # 3. Run the loop
    for profile in PROFILES:
        print_recommendations(profile["name"], profile["prefs"], songs, k=3, supabase=supabase, model=model)


if __name__ == "__main__":
    main()