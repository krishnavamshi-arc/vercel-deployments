import os
import re
import json
import requests
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "secretkey"

# Google Gemini API endpoint
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# ---------- Helpers ----------
def clean_text(text):
    text = re.sub(r"-\s+\n", "-", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s{2,}", " ", text).strip()

def read_pdf(file):
    reader = PdfReader(file)
    return clean_text(" ".join(page.extract_text() or "" for page in reader.pages))

def split_text(text, chunk_size=600, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_texts(texts, api_key):
    """Use Google Gemini embedding model"""
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    body = {"model": "embedding-001", "input": texts}
    r = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedText",
        headers=headers, json=body
    )
    data = r.json()
    return [np.array(e["embedding"], dtype=float) for e in data.get("embeddings", [])]

def get_top_chunks(query, chunks, embeddings, api_key, top_k=3):
    q_vec = embed_texts([query], api_key)[0].reshape(1, -1)
    sims = cosine_similarity(q_vec, np.vstack(embeddings))[0]
    top_ids = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_ids]

def generate_answer(contexts, query, api_key):
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    context_text = "\n\n".join(contexts)
    prompt = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer clearly and briefly."
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(API_URL, headers=headers, json=body)
    res = r.json()
    try:
        return res["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Could not generate answer. Please check your API key or input."

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf = request.files["pdf"]
        api_key = request.form["api_key"]
        if not pdf or not api_key:
            return "Missing PDF or API key"
        text = read_pdf(pdf)
        chunks = split_text(text)
        embeddings = embed_texts(chunks, api_key)
        session["chunks"] = chunks
        session["embeddings"] = [e.tolist() for e in embeddings]
        session["api_key"] = api_key
        return redirect(url_for("question"))
    return render_template("index.html")

@app.route("/question", methods=["GET", "POST"])
def question():
    if request.method == "POST":
        query = request.form["query"]
        api_key = session.get("api_key")
        chunks = session.get("chunks")
        embeddings = [np.array(e) for e in session.get("embeddings")]
        top_chunks = get_top_chunks(query, chunks, embeddings, api_key)
        answer = generate_answer(top_chunks, query, api_key)
        return render_template("answer.html", query=query, answer=answer, contexts=top_chunks)
    return render_template("question.html")

if __name__ == "__main__":
    app.run()
