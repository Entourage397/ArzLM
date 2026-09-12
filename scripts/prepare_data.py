import sys

from arzlm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare", *sys.argv[1:]]))
