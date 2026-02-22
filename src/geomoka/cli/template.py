from importlib.resources import files
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Scaffold minimal project folders")
    parser.add_argument("--root", type=str, default=None, help="Project root directory (default: current directory)")
    parser.add_argument("--config", type=str, default='generic', help="Templete config name options: generic (default), isprs_postdam")
    args = parser.parse_args()

    root = Path(args.root) if args.root else Path.cwd()
    dest_dir = root / 'config' / args.config
    dest_dir.mkdir(parents=True, exist_ok=True)

    src = files("geomoka").joinpath('template').joinpath('config').joinpath(args.config)
    for file in src.iterdir():
        if file.is_file():
            dest = dest_dir / file.name
            if not dest.exists():
                dest.write_bytes(file.read_bytes())

if __name__ == "__main__":
    main()