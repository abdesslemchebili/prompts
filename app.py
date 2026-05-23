import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------
# GROQ API KEY
# ---------------------------------------------------
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")

# ---------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

# GROQ MODEL
MODEL_NAME = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------
# GENERIC MODEL CALL
# ---------------------------------------------------
def _ask_model(messages):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
    )

    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------
# ERROR FORMATTER
# ---------------------------------------------------
def _format_api_error(exc: Exception) -> tuple[str, int]:

    message = str(exc)
    lowered = message.lower()

    # Rate limit
    if "429" in message or "rate limit" in lowered:

        return (
            "Rate limit reached for Groq API. Please wait and try again.",
            429,
        )

    # Authentication
    if "401" in message or "invalid api key" in lowered:

        return (
            "Invalid Groq API key.",
            401,
        )

    return message, 500


# ---------------------------------------------------
# ZERO SHOT
# ---------------------------------------------------
def zero_shot_response(user_prompt: str, category: str) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an expert AI assistant specialized in {category}. "
                f"Answer clearly and professionally."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return _ask_model(messages)


# ---------------------------------------------------
# FEW SHOT
# ---------------------------------------------------
def few_shot_response(user_prompt: str, category: str) -> str:

    messages = [

        {
            "role": "system",
            "content": (
                f"You are an expert AI assistant specialized in {category}. "
                f"Answer clearly and professionally."
            ),
        },

        # Example 1
        {
            "role": "user",
            "content": "Explain artificial intelligence simply.",
        },
        {
            "role": "assistant",
            "content": (
                "Artificial intelligence allows machines "
                "to simulate human intelligence."
            ),
        },

        # Example 2
        {
            "role": "user",
            "content": "Explain cloud computing.",
        },
        {
            "role": "assistant",
            "content": (
                "Cloud computing provides computing services "
                "through the internet."
            ),
        },

        # Real Prompt
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return _ask_model(messages)


# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------
# COMPARE ROUTE
# ---------------------------------------------------
@app.route("/compare", methods=["POST"])
def compare():

    data = request.get_json(silent=True) or {}

    prompt = data.get("prompt", "").strip()
    category = data.get("category", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    if not category:
        return jsonify({"error": "Category is required."}), 400

    try:

        zero_result = zero_shot_response(prompt, category)

        few_result = few_shot_response(prompt, category)

        return jsonify(
            {
                "prompt": prompt,
                "category": category,
                "zero_shot": zero_result,
                "few_shot": few_result,
            }
        )

    except Exception as exc:

        error_message, status_code = _format_api_error(exc)

        return jsonify({"error": error_message}), status_code


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)