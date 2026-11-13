"""Validering av den felles kalkulatoren mot FinQA dev.json.

1. Små enhetstester av clean_number, parse_arg, calculator_tool og loggen.
2. Kjører run_program på alle dev-eksempler som KUN bruker
   add/subtract/multiply/divide, og sammenligner med exe_ans.
   FinQA har rundet exe_ans til 5 desimaler, så et svar regnes som riktig når det
   blir likt fasiten avrundet til 5 desimaler (avvik på høyst 0.000005).

Del 2 bruker hele dev.json, som ikke ligger i repoet. Kjør fra prosjektroten:
    python scripts/last_ned_finqa.py dev
    python scripts/valider_kalkulator.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kalkulator import (  # noqa: E402
    OPS, calculator_tool, clean_number, get_log, parse_arg, reset_log,
    run_program,
)

DEV_PATH = ROOT / "data" / "raw" / "dev.json"
REL_TOL = 1e-4          # relativ toleranse (for store tall)
ABS_TOL = 0.5e-5        # avrunding til 5 desimaler, slik FinQA har rundet exe_ans
OP_RE = re.compile(r"([a-z_]+)\(")


def enhetstester():
    """Små tester. Kaster AssertionError ved første feil."""
    assert clean_number("5%") == 0.05
    assert clean_number("1,234") == 1234.0
    assert clean_number("$500") == 500.0
    assert clean_number(" $1,234.5 ") == 1234.5
    assert clean_number(7) == 7.0 and isinstance(clean_number(7), float)
    assert parse_arg("5%", []) == 0.05
    assert parse_arg("const_100", []) == 100.0
    assert parse_arg("const_m1", []) == -1.0
    assert parse_arg("#1", [10.0, 20.0]) == 20.0
    assert run_program("subtract(5829, 5735), divide(#0, 5735)") == (5829 - 5735) / 5735
    assert run_program("add(5%, 1)") == 1.05

    # run_program og calculate skal fortsatt kaste feil (for RAG-pipelinen)
    for bad in ["divide(1, 0)", "power(2, 3)", "add(abc, 1)"]:
        try:
            run_program(bad)
        except (ValueError, ZeroDivisionError):
            pass
        else:
            raise AssertionError(f"run_program kastet ikke feil på {bad!r}")

    # calculator_tool kaster aldri feil, og logger hvert kall
    reset_log()
    assert get_log() == []
    assert calculator_tool("multiply", "2", 3) == 6.0
    r = calculator_tool("divide", 1, 0)
    assert isinstance(r, str) and r.startswith("FEIL:"), r
    assert calculator_tool("power", 2, 3).startswith("FEIL:")
    assert calculator_tool("add", "abc", 1).startswith("FEIL:")
    log = get_log()
    assert len(log) == 4
    assert log[0]["op"] == "multiply" and log[0]["result"] == 6.0 and log[0]["error"] is None
    assert log[1]["error"].startswith("FEIL:") and log[1]["result"] is None
    assert all("timestamp" in e for e in log)
    reset_log()
    assert get_log() == []
    print("Enhetstester: alle OK")


def valider_mot_fasit():
    if not DEV_PATH.exists():
        sys.exit("data/raw/dev.json mangler. Kjør først: python scripts/last_ned_finqa.py dev")
    data = json.loads(DEV_PATH.read_text())
    utvalg = [e for e in data
              if set(OP_RE.findall(e["qa"]["program"])) <= set(OPS)]
    treff, bom = 0, []
    for e in utvalg:
        prog, fasit = e["qa"]["program"], e["qa"]["exe_ans"]
        try:
            svar = run_program(prog)
        except Exception as ex:
            bom.append((e["id"], prog, f"{type(ex).__name__}: {ex}", fasit))
            continue
        try:
            fasit_f = float(fasit)
        except (TypeError, ValueError):
            bom.append((e["id"], prog, svar, fasit))
            continue
        # Riktig hvis avviket skyldes avrundingen i FinQA (5 desimaler) eller er svært lite relativt.
        if abs(svar - fasit_f) <= max(ABS_TOL + 1e-12, REL_TOL * abs(fasit_f)):
            treff += 1
        else:
            bom.append((e["id"], prog, svar, fasit))

    print(f"\nDev-eksempler totalt: {len(data)}")
    print(f"Testet (kun add/subtract/multiply/divide): {len(utvalg)}")
    print(f"Treff: {treff}  ({treff / len(utvalg):.1%})")
    print(f"Bom:   {len(bom)}")
    for id_, prog, svar, fasit in bom:
        print(f"\n  {id_}\n    program: {prog}\n    vårt svar: {svar}\n    fasit:     {fasit}")
    return bom


if __name__ == "__main__":
    enhetstester()
    valider_mot_fasit()
