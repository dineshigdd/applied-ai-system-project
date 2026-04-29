import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import sys

# Load keys from the .env file
load_dotenv()
URL = os.getenv("SUPABASE_URL")
MASTER_KEY = os.getenv("SUPABASE_MASTER_KEY")

if not URL or not MASTER_KEY:
    print("ERROR: SUPABASE_URL and SUPABASE_MASTER_KEY not set in .env file")
    sys.exit(1)

# Create the client using the Master Key (Service Role)
print("Connecting to Supabase...")
try:
    supabase = create_client(URL, MASTER_KEY)
    print("✓ Connected to Supabase")
except Exception as e:
    print(f"ERROR: Failed to connect to Supabase: {e}")
    sys.exit(1)

# Load Jina model
print("Loading Jina embeddings model (this may take 5-15 minutes on first run)...")
sys.stdout.flush()
try:
    model = SentenceTransformer('jinaai/jina-embeddings-v5-text-small-retrieval', trust_remote_code=True)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    sys.exit(1)

# Read CSV
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'songs.csv')
if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at {csv_path}")
    sys.exit(1)

print(f"Reading CSV from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"✓ Read {len(df)} songs from CSV")

print(f"Uploading {len(df)} songs to Supabase...")
success_count = 0
error_count = 0

for index, row in df.iterrows():
    try:
        # We combine the features into one "concept" for the AI to understand
        text_to_vectorize = f"Genre: {row['genre']}, Mood: {row['mood']}, Description: {row['title']} by {row['artist']}"
        
        # Generate the 1024-dimension vector
        embedding = model.encode(text_to_vectorize).tolist()
        
        song_data = {
            "title": row['title'],
            "artist": row['artist'],
            "genre": row['genre'],
            "mood": row['mood'],
            "embedding": embedding
        }
        
        # Insert with error handling
        supabase.table("music_catalog").insert(song_data).execute()
        success_count += 1
        
        # Progress indicator every 5 songs
        if (index + 1) % 5 == 0:
            print(f"  Processed {index + 1}/{len(df)} songs...")
            
    except Exception as e:
        print(f"ERROR uploading song {index + 1} ({row.get('title', 'Unknown')}): {e}")
        error_count += 1

print(f"\n✓ Done! Uploaded {success_count}/{len(df)} songs successfully")
if error_count > 0:
    print(f"⚠ {error_count} songs failed to upload")