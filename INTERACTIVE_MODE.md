# 🎵 Interactive Music Recommender Guide

## How to Run

```bash
python3 -m src.main
```

## Modes

### 1. Batch Mode (Option 1)
Runs all predefined test profiles automatically:
- Gym Session (energetic pop)
- Late Night Study (lofi, focused)
- Sunday Morning (bossa nova, dreamy)
- Road Trip (rock, energetic)
- And edge case profiles...

### 2. Interactive Mode (Option 2)
**Describe what you want to hear in plain language!**

The system will parse your input and recommend songs based on:
- **Genre**: pop, rock, jazz, lofi, hip-hop, soul, reggae, latin, bossa nova, blues, folk, electronic, trap, indie, country
- **Mood**: happy, sad, energetic, relaxed, focused, dreamy, uplifting, lonely, intense
- **Energy**: automatically inferred from context (workout = high, study = low, etc.)

## Examples

```
🎤 What kind of music do you want? > happy pop music for a workout
   → Detects: genre=pop, mood=happy, energy=0.85

🎤 What kind of music do you want? > chill lofi for studying
   → Detects: genre=lofi, mood=relaxed, energy=0.35

🎤 What kind of music do you want? > energetic electronic music for a party
   → Detects: genre=electronic, mood=energetic, energy=0.85

🎤 What kind of music do you want? > sad blues
   → Detects: genre=blues, mood=sad, energy=0.5 (default)
```

## How Parsing Works

1. **Keyword Matching**: The system scans your input for genre and mood keywords
2. **Energy Inference**: Looks for context clues:
   - High energy (0.85): "workout", "gym", "party", "dance", "exercise"
   - Low energy (0.35): "study", "sleep", "relax", "focus", "chill"
   - Neutral (0.5): default if no context detected
3. **Preference Dict**: Builds a structured preference object for the recommender

## Features

✅ Natural language parsing (no structured forms needed)
✅ Multiple recommendations per query
✅ Loop for multiple queries in one session
✅ Clear feedback on what was understood
✅ Graceful error handling
✅ Fallback to lightweight embedding model if needed
