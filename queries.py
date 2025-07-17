import os
import uuid
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pdfplumber
from docx import Document
import os
import json
import csv
import xml.etree.ElementTree as ET
from openpyxl import load_workbook
import pandas as pd
from dotenv import load_dotenv, dotenv_values 

load_dotenv()

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = QdrantClient(host="localhost", port=6333)
collection_name = "battery_chunks"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

query = "What type of battery is BAT-0002?"
query_vector = model.encode(query)

response = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=20
)

retrieved_chunks = [point.payload.get("text") for point in response.points if point.payload.get("text")]
context = "\n\n".join(retrieved_chunks)

def generate_prompt(context: str, query: str) -> str:
    return f"""You are a technical expert.

Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""

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
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

prompt = generate_prompt(context, query)
answer = ask_groq_llm(prompt, model=GROQ_MODEL, key=GROQ_API_KEY)
print(f"\nAnswer: {answer}\n")
for point in response.points:
    score = point.score
    file = point.payload.get("file", "Unknown")
    print(f"Score: {score:.4f}\nFile: {file}\n")