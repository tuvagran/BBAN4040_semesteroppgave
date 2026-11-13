"""Last ned hele FinQA (dev.json, test.json, train.json) til data/raw/.

Repoet inneholder bare rapportsidene utvalgene trenger (data/finqa_utvalg/).
Hele datasettet trengs bare for to ting:
- å trekke dev og val på nytt:     python -m src.utvalg
- å teste kalkulatoren på hele dev: python scripts/valider_kalkulator.py

Kjør fra prosjektroten:
    python scripts/last_ned_finqa.py            # alle tre filene (ca. 100 MB)
    python scripts/last_ned_finqa.py dev test   # bare noen av dem

Kilde: https://github.com/czyssrs/FinQA (Chen et al., 2021, MIT-lisens, se data/raw/LICENSE_FinQA).
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
SPLITS = ("dev", "test", "train")


def last_ned(split: str) -> None:
    path = DATA_RAW / f"{split}.json"
    if path.exists():
        print(f"{path.relative_to(ROOT)} finnes allerede, hopper over")
        return
    print(f"Laster ned {split}.json ...", end=" ", flush=True)
    tmp = path.with_suffix(".json.part")          # halvferdige filer blir aldri liggende som .json
    urllib.request.urlretrieve(URL.format(split=split), tmp)
    tmp.rename(path)
    print(f"{path.stat().st_size / 1e6:.0f} MB -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    valgte = sys.argv[1:] or SPLITS
    ukjente = set(valgte) - set(SPLITS)
    if ukjente:
        sys.exit(f"Ukjent split: {sorted(ukjente)}. Velg blant {SPLITS}.")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for split in valgte:
        last_ned(split)
