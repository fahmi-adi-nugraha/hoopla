import os
import sys

from augmented_generation.general import run
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        print("Missing API key for LLM")
        sys.exit(1)

    try:
        run(api_key)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
