"""Felles utvalg av spørsmål for RAG-pipelinen og agenten.

Utvalgene er lagret som filer som begge leser:
    from src.utvalg import load_utvalg, load_examples
    dev = load_utvalg("dev")                # DataFrame med id, question, gold_answer, ...
    examples = load_examples("dev")         # id -> hele FinQA-eksempelet

Rapportsidene ligger i data/finqa_utvalg/ (dev.json, val.json, test.json). Det er bare
sidene til spørsmålene i utvalgene, ikke hele FinQA. load_examples leser derfra.

Dev og val kan trekkes på nytt og blir identiske (fast seed). Det krever hele FinQA:
    python scripts/last_ned_finqa.py        # laster ned dev.json, test.json og train.json til data/raw/
    python -m src.utvalg                    # lager data/utvalg_{dev,val}.csv og data/finqa_utvalg/{dev,val}.json

Testutvalget (data/utvalg_test.csv) er fast: det er de 200 spørsmålene testkjøringen ble
gjort på, og det trekkes ikke på nytt. Det ble trukket med seed 42 blant testspørsmålene
som ikke står i data/ekskludert_fra_test.csv (100 spørsmål fra en tidlig utprøving av RAG-pipelinen).

Regler:
- Bare spørsmål der qa.program kun bruker add, subtract, multiply, divide.
- Fasit = qa.exe_ans.
- dev: N_DEV fra dev.json, seed SEED. Brukes til å utvikle promptene.
- val: N_VAL fra train.json, seed SEED. Brukes til å teste designvalg på andre
  spørsmål enn dem promptene ble utviklet på. Ingen modeller trenes på train.json.
- test: N_TEST fra test.json. Kjøres én gang, når begge systemene er ferdige.
"""
import sys
import json
import re

import pandas as pd

from src.config import (ALLOWED_OPS, DATA_RAW, FINQA_UTVALG, N_DEV, N_VAL, ROOT,
                        SEED, UTVALG_DEV, UTVALG_TEST, UTVALG_VAL)

_OP_RE = re.compile(r"([a-z_]+)\(")
_PATHS = {"dev": UTVALG_DEV, "test": UTVALG_TEST, "val": UTVALG_VAL}
_SOURCE = {"dev": "dev", "test": "test", "val": "train"}     # hvilken FinQA-fil hvert utvalg kommer fra
_SIZE = {"dev": N_DEV, "val": N_VAL}                        # utvalgene som kan trekkes på nytt


def load_examples(split: str) -> dict:
    """Rapportsidene til spørsmålene i utvalget, som oppslag fra id til FinQA-eksempel.

    Leser data/finqa_utvalg/{split}.json, som ligger i repoet. Hele FinQA trengs ikke.
    """
    data = json.loads((FINQA_UTVALG / f"{split}.json").read_text(encoding="utf-8"))
    return {e["id"]: e for e in data}


def load_full(split: str) -> dict:
    """ALLE FinQA-eksempler i en split (data/raw/, lastes ned med scripts/last_ned_finqa.py).

    Trengs bare for å trekke dev og val på nytt. "val" leser train.json.
    """
    path = DATA_RAW / f"{_SOURCE[split]}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} mangler. Kjør: python scripts/last_ned_finqa.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {e["id"]: e for e in data}


def save_examples(split: str, examples: dict) -> None:
    """Lagre rapportsidene til spørsmålene i utvalget i data/finqa_utvalg/{split}.json."""
    ids = load_utvalg(split)["id"]
    FINQA_UTVALG.mkdir(parents=True, exist_ok=True)
    (FINQA_UTVALG / f"{split}.json").write_text(
        json.dumps([examples[i] for i in ids], ensure_ascii=False), encoding="utf-8")


def _rows(examples: dict) -> pd.DataFrame:
    rows = []
    for e in examples.values():
        ops = _OP_RE.findall(e["qa"]["program"])
        rows.append({
            "id": e["id"],
            "question": e["qa"]["question"],
            "gold_answer": e["qa"]["exe_ans"],
            "gold_program": e["qa"]["program"],
            "n_steps": len(ops),
            "gold_inds": sorted(e["qa"]["gold_inds"]),
            "kalkulator_ok": set(ops) <= set(ALLOWED_OPS),
        })
    return pd.DataFrame(rows)


def make_utvalg(splits: tuple = ("dev", "val")) -> dict:
    """Trekk og lagre utvalgene i `splits` (dev og/eller val). Returnerer en oversikt."""
    if "test" in splits:
        raise ValueError("Testutvalget er fast (data/utvalg_test.csv) og skal ikke trekkes på nytt.")
    oversikt = {}
    for split in splits:
        full = load_full(split)
        alle = _rows(full)
        pool = alle[alle["kalkulator_ok"]]
        utvalg = pool.sample(n=_SIZE[split], random_state=SEED).drop(columns="kalkulator_ok").reset_index(drop=True)
        out = utvalg.copy()
        out["gold_inds"] = out["gold_inds"].map(json.dumps)
        out.to_csv(_PATHS[split], index=False)
        save_examples(split, full)                  # bare rapportsidene til de trukne spørsmålene
        oversikt[split] = {"alle": len(alle), "kun_4_regnearter": len(pool), "trukket": len(utvalg),
                           "fil": str(_PATHS[split].relative_to(ROOT))}
    return oversikt


def load_utvalg(split: str) -> pd.DataFrame:
    """Les et lagret utvalg. gold_inds kommer tilbake som liste."""
    df = pd.read_csv(_PATHS[split])
    df["gold_inds"] = df["gold_inds"].map(json.loads)
    return df


if __name__ == "__main__":
    for split, info in make_utvalg(tuple(sys.argv[1:]) or ("dev", "val")).items():
        print(split, info)
