import mimetypes
from pathlib import Path

from describe_image.opts import get_opts
from google import genai
from google.genai.types import Part

DATA_DIR = "data"


def run(api_key: str) -> None:
    cli_opts, parser = get_opts()

    mime, _ = mimetypes.guess_type(cli_opts.image)
    mime = mime or "image/jpeg"

    with open(Path(DATA_DIR, "paddington.jpeg"), "rb") as img_file:
        img = img_file.read()

    client = genai.Client(api_key=api_key)

    prompt = """
    Given the included image and text query, rewrite the text query to improve search
    results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """

    parts = [
        prompt,
        Part.from_bytes(data=img, mime_type=mime),
        cli_opts.query.strip(),
    ]

    response = client.models.generate_content(model="gemini-2.5-flash", contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens: {response.usage_metadata.total_token_count}")
