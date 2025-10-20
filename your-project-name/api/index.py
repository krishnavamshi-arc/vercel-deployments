from flask import Flask, render_template, request
import os
import re
import requests
from PyPDF2 import PdfReader

app = Flask(__name__)

# ----------------------------
# Helper Functions
# ----------------------------
def clean_text(text: str) -> str:
    """Remove extra whitespace and fix line breaks."""
    if not text:
        return ""
    text = re.sub(r"-\s*\n", "-", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def read_pdf(file) -> str:
    """Extract text from uploaded PDF file."""
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return clean_text(" ".join(pages))

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf_file = request.files.get("pdf")
        question = request.form.get("question")
        api_key = request.form.get("api_key")

        if not pdf_file or not question or not api_key:
            return render_template("index.html", error="Please upload a PDF, enter a question, and provide your API key.")

        try:
            text = read_pdf(pdf_file)
            if not text:
                return render_template("index.html", error="Could not extract text from the PDF.")

            # Build prompt
            prompt = f"{question}\n\nUse the following document text as context:\n{text[:3000]}"

            # Google Gemini API call
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": "AIzaSyAAlV9eAGcg4yShhU0o6CE0-cFKPV8FsnY"
            }
            body = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=body)

            if response.status_code != 200:
                return render_template("index.html", error=f"API error: {response.text}")

            data = response.json()

            # Extract answer safely
            answer = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "No answer returned from API.")
            )

            return render_template("answer.html", question=question, answer=answer)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run()
