"""Create a source-only DJN release archive with security-focused exclusions."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import os

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_ROOT.parent / "DJN-revised-audited-source.zip"
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "node_modules",
    "staticfiles", "media", "logs", "outputs", "reports",
}
EXCLUDED_FILES = {
    "db.sqlite3", "credentials.json", "service-account.json", "temp.py",
    OUTPUT_FILE.name,
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".key", ".pem", ".p12", ".pfx", ".sqlite3", ".mp4"}


def excluded(path: Path) -> bool:
    name = path.name
    return (
        path.is_symlink()
        or name in EXCLUDED_FILES
        or path.suffix.lower() in EXCLUDED_SUFFIXES
        or name == ".env"
        or (name.startswith(".env.") and name != ".env.example")
    )


def main() -> None:
    count = 0
    with ZipFile(OUTPUT_FILE, "w", compression=ZIP_DEFLATED) as archive:
        for current, directories, files in os.walk(PROJECT_ROOT):
            base = Path(current)
            directories[:] = [
                name for name in directories
                if name not in EXCLUDED_DIRECTORIES and not (base / name).is_symlink()
            ]
            for name in files:
                source = base / name
                if excluded(source):
                    continue
                target = Path(PROJECT_ROOT.name) / source.relative_to(PROJECT_ROOT)
                archive.write(source, target)
                count += 1
    print(f"Created {OUTPUT_FILE} with {count} source files.")


if __name__ == "__main__":
    main()
