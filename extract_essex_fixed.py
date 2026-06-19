import io
import shutil
import zipfile
from pathlib import Path


OUTER_ZIP = Path("Face Recognition Data.zip")
OUTPUT_DIR = Path("data/raw/essex_fixed")
INNER_ZIPS = ("faces94.zip", "faces95.zip", "faces96.zip", "grimace.zip")


def safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise ValueError(f"Unsafe zip member path: {member_name}")
    return target


def extract_inner_zip(inner_name: str, payload: bytes, output_dir: Path) -> dict:
    subset = Path(inner_name).stem
    subset_dir = output_dir / subset
    files = 0
    dirs = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        for info in inner.infolist():
            target = safe_target(subset_dir, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                dirs += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with inner.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            files += 1
    return {"subset": subset, "files": files, "dirs": dirs}


def main() -> None:
    if not OUTER_ZIP.exists():
        raise FileNotFoundError(OUTER_ZIP)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    summary = []
    with zipfile.ZipFile(OUTER_ZIP) as outer:
        available = set(outer.namelist())
        missing = [name for name in INNER_ZIPS if name not in available]
        if missing:
            raise FileNotFoundError(f"Missing inner zips in {OUTER_ZIP}: {missing}")
        for inner_name in INNER_ZIPS:
            payload = outer.read(inner_name)
            summary.append(extract_inner_zip(inner_name, payload, OUTPUT_DIR))

    for row in summary:
        print(f"{row['subset']}: files={row['files']} dirs={row['dirs']}")
    print(f"Extracted to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
