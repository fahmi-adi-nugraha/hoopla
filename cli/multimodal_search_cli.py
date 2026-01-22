import sys

from multimodal_search.general import run


def main() -> None:
    try:
        run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
