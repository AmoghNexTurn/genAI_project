import os
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

QDRANT_URL = "https://1ae2c6f5-15a4-49e9-af8d-96ab7348e31e.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

collection_name = "battery_chunks"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

st.set_page_config(page_title="Battery Q&A", layout="wide")
st.title("🔋 Battery Document Q&A")

query = st.text_input("Enter your question:", placeholder="e.g., What type of battery is BAT-0002?")

# Load prompt template from file
def load_prompt_template(path="prompt_template.txt"):
    with open(path, "r") as file:
        return file.read()

# Fill in the template with actual values
def generate_prompt(context: str, query: str) -> str:
    template = load_prompt_template()
    return template.format(context=context, query=query)

def ask_groq_llm(prompt, model=GROQ_MODEL, key=GROQ_API_KEY):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a technical expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

if query:
    with st.spinner("🔍 Searching for relevant context..."):
        query_vector = model.encode(query)

        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=20
        )

        retrieved_chunks = [
            {"text": point.payload.get("text"), "file": point.payload.get("file")}
            for point in response.points
            if point.payload.get("text")
        ]
        context = "\n\n".join(f"[{chunk['file']}]\n{chunk['text']}" for chunk in retrieved_chunks)

        prompt = generate_prompt(context, query)

    with st.spinner("💬 Generating answer..."):
        try:
            answer = ask_groq_llm(prompt)
            st.success("✅ Answer generated:")
            st.markdown(f"**Answer:**\n\n{answer}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    st.subheader("📄 Top Matching Chunks")
    for i, point in enumerate(response.points):
        score = point.score
        text = point.payload.get("text", "[No text found]")
        file = point.payload.get("file", "Unknown file")
        with st.expander(f"Chunk #{i+1} (Score: {score:.4f}) from {file}"):
            st.markdown(text)
