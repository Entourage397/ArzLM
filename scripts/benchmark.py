import sys

from arzlm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["bench", *sys.argv[1:]]))
