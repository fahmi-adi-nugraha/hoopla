import os

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")

    prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

    print(f"PROMPT: {prompt}")
    print(f"RESPONSE: {response.text}")
    print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")


if __name__ == "__main__":
    main()
