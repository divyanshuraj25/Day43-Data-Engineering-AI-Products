
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY .env file mein nahi mili")

client = genai.Client(api_key=api_key)

try:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="FAISS knowledge base for an AI study assistant"
    )

    print("Gemini embedding successful!")
    print("Vector dimensions:", len(result.embeddings[0].values))

except Exception as e:
    print("Embedding error:", e)
