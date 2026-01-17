import sys

from evaluation.general import run
from evaluation.opts import get_opts


def main():
    cli_opts, parser = get_opts()

    try:
        run(cli_opts, parser)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
