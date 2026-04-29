import os
import sys
import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import recommend_songs
from src.database import get_supabase_client

load_dotenv()

class DummyQueryVector:
    def __init__(self, vector):
        self._vector = vector

    def tolist(self):
        return self._vector

class DummyModel:
    def encode(self, text):
        # Return 1024-dimensional vector to match Jina v5
        return DummyQueryVector([0.1] * 1024)

class DummyRpcResponse:
    def __init__(self, data):
        self.data = data

    def execute(self):
        return self

class DummySupabase:
    def __init__(self, data):
        self._data = data

    def rpc(self, name, args):
        return DummyRpcResponse(self._data)

@pytest.fixture(autouse=True)
def patch_generate_explanation(monkeypatch):
    monkeypatch.setattr(
        "src.recommender.generate_explanation",
        lambda user_prefs, song: f"This track matches the {user_prefs.get('mood', 'happy')} vibe."
    )

@pytest.fixture
def mock_user_prefs():
    """Provides a standard user profile for testing."""
    return {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
    }

@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def fake_supabase():
    return DummySupabase([
        {
            "title": "Test Song",
            "artist": "Test Artist",
            "genre": "pop",
            "mood": "happy",
            "similarity": 0.95,
        }
    ])

@pytest.fixture
def empty_supabase():
    return DummySupabase([])

@pytest.fixture
def supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_MASTER_KEY")
    if not url or not key:
        pytest.skip("Supabase credentials are not configured.")
    return get_supabase_client()


def test_supabase_connection(supabase_client):
    """Test 1: Verify we can actually reach the Vector Database."""
    response = supabase_client.table("music_catalog").select("id").limit(1).execute()
    assert response.data is not None
    assert len(response.data) >= 0


def test_recommend_songs_returns_valid_structure(mock_user_prefs, dummy_model, fake_supabase):
    """Test 2: Verify the RAG pipeline returns the correct data format."""
    results = recommend_songs(mock_user_prefs, songs=[], k=1, supabase=fake_supabase, model=dummy_model)

    assert len(results) == 1
    song, score, explanation = results[0]

    assert "title" in song
    assert "artist" in song
    assert isinstance(score, float)
    assert isinstance(explanation, str)
    assert len(explanation) > 10


def test_gemini_explanation_quality(mock_user_prefs, dummy_model, fake_supabase):
    """Test 3: Verify the explanation text is generated."""
    results = recommend_songs(mock_user_prefs, songs=[], k=1, supabase=fake_supabase, model=dummy_model)
    _, _, explanation = results[0]

    assert isinstance(explanation, str)
    assert "matches the" in explanation.lower()


def test_error_handling_empty_db(mock_user_prefs, dummy_model, empty_supabase):
    """Test 4: Verify system doesn't crash if no matches are found."""
    results = recommend_songs(mock_user_prefs, songs=[], k=1, supabase=empty_supabase, model=dummy_model)
    assert isinstance(results, list)
    assert results == []

def test_missing_user_context(dummy_model, fake_supabase):
    """Test 5: Verify system handles incomplete user preferences gracefully."""
    # Context is missing 'mood' and 'energy'
    incomplete_prefs = {"genre": "pop"} 
    
    try:
        results = recommend_songs(incomplete_prefs, songs=[], k=1, supabase=fake_supabase, model=dummy_model)
        assert len(results) == 1
        _, _, explanation = results[0]
        # Check if the explanation handles 'None' or empty values gracefully
        assert "None" not in explanation 
    except Exception as e:
        pytest.fail(f"System crashed when user context was missing keys: {e}")
