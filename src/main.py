import os
from supabase import create_client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re
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

# --- SUPPORTED GENRES AND MOODS ---
GENRE_KEYWORDS = {
    "pop": ["pop"],
    "rock": ["rock"],
    "jazz": ["jazz"],
    "lofi": ["lofi", "lo-fi", "chill hop"],
    "hip-hop": ["hip-hop", "hip hop", "rap"],
    "soul": ["soul", "r&b"],
    "reggae": ["reggae"],
    "latin": ["latin"],
    "bossa nova": ["bossa", "bossa nova"],
    "blues": ["blues"],
    "folk": ["folk"],
    "electronic": ["electronic", "edm"],
    "trap": ["trap"],
    "indie": ["indie"],
    "country": ["country"],
}

MOOD_KEYWORDS = {
    "happy": ["happy", "cheerful", "joyful", "upbeat", "positive"],
    "sad": ["sad", "melancholic", "downbeat"],
    "energetic": ["energetic", "energized", "pumped"],
    "relaxed": ["relaxed", "chill", "laid-back"],
    "focused": ["focused", "concentration", "study"],
    "dreamy": ["dreamy", "ethereal"],
    "uplifting": ["uplifting"],
    "lonely": ["lonely"],
    "intense": ["intense"],
}

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


def load_embedding_model():
    """Load embedding model with fallback to lighter model if memory is insufficient."""
    models = [
        ('jinaai/jina-embeddings-v5-text-small-retrieval', 'Jina v5 (Primary)'),
        ('sentence-transformers/stsb-roberta-large', 'STSB RoBERTa Large (1024-dim fallback)'),
    ]
    
    for model_name, label in models:
        try:
            print(f"Loading AI Model ({label})... this may take a moment...")
            model = SentenceTransformer(model_name, trust_remote_code=True)
            print(f"✓ Loaded {label} successfully")
            return model
        except OSError as e:
            if "paging file" in str(e).lower() or "memory" in str(e).lower():
                print(f"⚠ Insufficient memory for {label}. Trying lighter alternative...")
                continue
            else:
                raise
    
    print("ERROR: Could not load any embedding model. Check your system memory.")
    raise RuntimeError("Failed to load embedding model after trying all options.")


def parse_user_input(user_input: str) -> dict:
    """
    Parse natural language input to extract genre, mood, and energy.
    Example: "I want happy pop music for a workout" 
    → {"genre": "pop", "mood": "happy", "energy": 0.85}
    """
    text = user_input.lower().strip()
    
    # Infer energy from context
    energy_keywords_high = ["workout", "gym", "exercise", "running", "party", "dance", "festival", "club"]
    energy_keywords_low = ["sleep", "bedtime", "night", "chill", "relax", "study", "focus", "work"]
    
    # Extract genre
    detected_genre = None
    for genre, keywords in GENRE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            detected_genre = genre
            break
    
    # Extract mood
    detected_mood = None
    for mood, keywords in MOOD_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            detected_mood = mood
            break
    
    # Infer energy
    energy = 0.5  # default
    if any(keyword in text for keyword in energy_keywords_high):
        energy = 0.85
    elif any(keyword in text for keyword in energy_keywords_low):
        energy = 0.35
    
    # Build preference dict
    prefs = {}
    if detected_genre:
        prefs["genre"] = detected_genre
    if detected_mood:
        prefs["mood"] = detected_mood
    prefs["energy"] = energy
    
    return prefs


def interactive_mode(supabase, model, songs: list) -> None:
    """
    Interactive loop where user enters music preferences and gets recommendations.
    """
    print("\n" + "=" * 60)
    print(" 🎵 INTERACTIVE MODE - Describe Your Music Vibe")
    print("=" * 60)
    print("\nAvailable Genres:")
    print("  " + ", ".join(GENRE_KEYWORDS.keys()))
    print("\nAvailable Moods:")
    print("  " + ", ".join(MOOD_KEYWORDS.keys()))
    print("\nExamples:")
    print("  • 'happy pop music for a workout'")
    print("  • 'chill lofi for studying'")
    print("  • 'energetic music for a party'")
    print("\nCommands:")
    print("  • 'genres' - Show all genres")
    print("  • 'moods' - Show all moods")
    print("  • 'quit' - Exit interactive mode")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("🎤 What kind of music do you want? > ").strip()
        
        # Handle special commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Exiting interactive mode. Goodbye!")
            break
        
        if user_input.lower() == "genres":
            print("\n📚 Available Genres:")
            for i, genre in enumerate(GENRE_KEYWORDS.keys(), 1):
                print(f"   {i:2d}. {genre}")
            print()
            continue
        
        if user_input.lower() == "moods":
            print("\n💭 Available Moods:")
            for i, mood in enumerate(MOOD_KEYWORDS.keys(), 1):
                print(f"   {i:2d}. {mood}")
            print()
            continue
        
        if not user_input:
            print("Please enter a music preference.\n")
            continue
        
        # Parse user input
        prefs = parse_user_input(user_input)
        
        if not prefs or (len(prefs) == 1 and "energy" in prefs):
            print("⚠ Could not understand your preference. Please mention a genre or mood.")
            print("   Type 'genres' or 'moods' to see available options.\n")
            continue
        
        # Show what we understood
        print(f"\n📍 You asked for: {user_input}")
        print(f"   Genre: {prefs.get('genre', 'Any')}")
        print(f"   Mood: {prefs.get('mood', 'Any')}")
        print(f"   Energy: {prefs.get('energy', 'Balanced')}")
        
        # Get recommendations
        print("\n⏳ Finding recommendations...")
        try:
            print_recommendations(
                profile_name="Your Request",
                user_prefs=prefs,
                songs=songs,
                k=3,
                supabase=supabase,
                model=model
            )
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}\n")
        
        # Ask if user wants more
        again = input("\n🔄 Want another recommendation? (yes/no): ").strip().lower()
        if again not in ["yes", "y"]:
            break
    
    print()


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
    
    model = load_embedding_model()
    
    # 2. Load songs (still useful for local reference if needed)
    songs = load_songs("data/songs.csv")

    # 3. Menu: Choose between batch and interactive modes
    print("\n" + "=" * 60)
    print(" 🎵 Music Recommender System")
    print("=" * 60)
    print("\n1. 📊 Batch Mode - Run all test profiles")
    print("2. 💬 Interactive Mode - Describe what you want to hear")
    print("\n" + "=" * 60)
    
    choice = input("\nChoose mode (1 or 2): ").strip()
    
    if choice == "1":
        # Batch mode: run all profiles
        print("\n🚀 Running test profiles...\n")
        for profile in PROFILES:
            print_recommendations(profile["name"], profile["prefs"], songs, k=3, supabase=supabase, model=model)
    elif choice == "2":
        # Interactive mode
        interactive_mode(supabase, model, songs)
    else:
        print("❌ Invalid choice. Exiting.")


if __name__ == "__main__":
    main()