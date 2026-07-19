import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


def generate_learning_content(topic):

    if not API_KEY:
        return "Error: GEMINI_API_KEY not found in .env file."

    try:

        prompt = f"""
You are an expert teacher.

Explain the following topic in simple English for students.

Topic:
{topic}

Include:
1. Introduction
2. Explanation
3. Example
4. Advantages
5. Summary
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Error: {e}"