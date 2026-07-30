"""Format this project's Typst sources with the agreed Typstyle settings."""

from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parent
COMMAND = [
    "typstyle",
    "--line-width",
    "100",
    "--wrap-text",
    "--inplace",
    ".",
]


def main() -> None:
    subprocess.run(COMMAND, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
