# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**CoreAudio RAG Engine 1.0**  

---

### Reflection and Ethics
#### System Limitations & Biases
The primary limitation is data bias: my system only knows what is in the CSV/database labels. If the human-assigned labels are wrong or narrow, the AI's "Why" explanation might be hallucinated to justify a bad match. Additionally, because I use the Jina v5 embedding model, the system might have a "Western bias" and struggle to correctly rank traditional or indie music from other cultures that aren't well-represented in the model's training data.

#### Potential Misuse & Prevention
The AI could be misused to manipulate user behavior—for example, a developer could secretly weight specific artists higher to get more "clicks" while the AI "fakes" a convincing reason why it fits the user’s vibe. To prevent this, I would implement Explainable AI (XAI) logs, where the raw similarity scores are saved alongside the generated explanations so that a human auditor can verify the math actually matches the words.

#### Reliability Surprise
During testing, I was surprised by how much better the AI was at handling conflicting moods than my old math system. In my original project, a "Sad" mood and "High Energy" would cancel each other out and give a low score. In this system, the AI actually found a "Bittersweet" track and explained that the high energy was a "mask" for the sad lyrics, which felt much more human.

### Collaboration with AI
Throughout this project, I used AI as a pair-programmer to bridge the gap between my software development background and machine learning.
**- Helpful Suggestion:** The AI suggested using the .get() method in my main.py print loop. This was a "Senior Dev" move that prevented the entire program from crashing when I tested a profile against a song that was missing its "Mood" tag in Supabase.

**- Flawed Suggestion:** At one point, the AI suggested a complex mathematical normalization for the energy score that required a column I hadn't created yet. It assumed my database structure was more complex than it actually was, which would have led to a KeyError if I hadn't caught it. This reminded me that I always need to verify if the AI’s logic matches my actual system architecture.

### Reflection: Final Summary
**- From Math to Meaning:**
 I learned that while my old system was good at math, AI is better at "vibes." Moving to RAG taught me how to handle human context that hard-coded rules can't capture.
**-Data Orchestration:** 
I realized that AI problem-solving is about building a pipeline. It’s not just about the code; it's about making the database, the embeddings, and the LLM talk to each other correctly.
**-User-Centric Design:**
 Hiding the "Energy" score taught me that users prefer natural explanations over raw numbers. I learned to use AI as a translator between complex data and human-friendly results.
