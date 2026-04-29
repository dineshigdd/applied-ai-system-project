import runpy
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sync_music.py"
    if not script_path.exists():
        raise FileNotFoundError(f"sync_music script not found at {script_path}")
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
