import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI


load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Please set it in your environment.")

# Gemini provides an OpenAI-compatible endpoint.
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")


def _classify_with_chat_completion(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a news classifier. Return exactly one label: Sport, Tech, Politique, or Sante.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = (response.choices[0].message.content or "").strip()
    # Keep output predictable for the UI.
    normalized = content.lower()
    if "sport" in normalized:
        return "Sport"
    if "tech" in normalized:
        return "Tech"
    if "politique" in normalized:
        return "Politique"
    if "sante" in normalized or "santé" in normalized:
        return "Sante"
    return content or "Unknown"


def _format_api_error(exc: Exception) -> tuple[str, int]:
    message = str(exc)
    lowered = message.lower()

    # Gemini free tier rate limit/quota exceeded.
    if "429" in message or "quota" in lowered or "resource_exhausted" in lowered:
        retry_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", lowered)
        retry_note = ""
        if retry_match:
            seconds = int(float(retry_match.group(1)))
            retry_note = f" Retry in about {seconds} seconds."

        friendly = (
            "Rate limit reached for Gemini API."
            " You exceeded your current quota for this model."
            f"{retry_note} You can wait, switch model, or increase quota."
        )
        return friendly, 429

    return message, 500


def classify_email_zero_shot(email_text: str) -> str:
    prompt = (
        "Classify the news text as Sport, Tech, Politique, or Sante.\n"
        "Return only one label: Sport, Tech, Politique, or Sante.\n\n"
        f"Text: {email_text}"
    )

    return _classify_with_chat_completion(prompt)


def classify_email_few_shot(email_text: str) -> str:
    prompt = (
        "Classify the news text as Sport, Tech, Politique, or Sante.\n"
        "Examples:\n"
        "Text: The team scored twice in the final minutes to win the championship.\n"
        "Label: Sport\n\n"
        "Text: The startup launched a new AI chip for mobile devices.\n"
        "Label: Tech\n\n"
        "Text: Parliament approved the new tax reform after a long debate.\n"
        "Label: Politique\n\n"
        "Text: Doctors recommend daily exercise to reduce heart disease risk.\n"
        "Label: Sante\n\n"
        "Now classify this text and return only one label (Sport, Tech, Politique, or Sante).\n"
        f"Text: {email_text}\n"
        "Label:"
    )

    return _classify_with_chat_completion(prompt)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/zero-shot", methods=["POST"])
def zero_shot():
    data = request.get_json(silent=True) or {}
    email_text = data.get("email", "").strip()

    if not email_text:
        return jsonify({"error": "Text is required."}), 400

    try:
        label = classify_email_zero_shot(email_text)
        return jsonify({"method": "zero-shot", "label": label})
    except Exception as exc:
        error_message, status_code = _format_api_error(exc)
        return jsonify({"error": error_message}), status_code


@app.route("/few-shot", methods=["POST"])
def few_shot():
    data = request.get_json(silent=True) or {}
    email_text = data.get("email", "").strip()

    if not email_text:
        return jsonify({"error": "Text is required."}), 400

    try:
        label = classify_email_few_shot(email_text)
        return jsonify({"method": "few-shot", "label": label})
    except Exception as exc:
        error_message, status_code = _format_api_error(exc)
        return jsonify({"error": error_message}), status_code


if __name__ == "__main__":
    app.run(debug=True)
