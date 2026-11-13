"""Felles korrekthetsregler for RAG-pipelinen og agenten.

Begge betingelsene importerer herfra, slik at svarene rettes likt:
    from src.evaluering import parse_answer, is_correct, is_correct_rounded, is_scale_error

Regler (avtalt):
- Hovedregel (`is_correct`): relativ feil < REL_TOL (1 %) mot qa.exe_ans.
- Prosent normaliseres: tekstsvaret "14%" tolkes som 0.14 (`parse_answer`).
  Fasiten (exe_ans) og kalkulatoren gir andeler, så tall sammenlignes andel mot andel.
- Tilleggsmål (`is_correct_rounded`): like etter avrunding til DECIMALS (3) desimaler.
  Brukes bare som robusthetssjekk. Det godtar alle svar under 0.0005 som riktige.
- Skalafeil (`is_scale_error`): feil nå, men riktig delt eller ganget med 100.
  Brukes til failure_stage = "formatering".

Avrunding (tilleggsmålet): skoleavrunding med desimalaritmetikk. Pythons round() på
float kan runde "samme" tall ulikt (1.2465 lagres som 1.24649999... eller 1.24650000...),
så vi runder tallets korteste desimalrepresentasjon (repr).
"""
import re
from decimal import Decimal, ROUND_HALF_UP

from src.config import DECIMALS, REL_TOL

_NUMBER_RE = re.compile(r"[-−]?\(?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?\s*%?")


def parse_answer(raw):
    """Gjør et svar (tall eller tekst) om til en float i desimalform. None hvis det ikke går.

    0.124          -> 0.124
    "0.124"        -> 0.124
    "14%"          -> 0.14     (prosent normaliseres til andel)
    "$1,234.5"     -> 1234.5
    "(32)"         -> -32.0    (regnskapsnotasjon for negative tall)
    "Answer: -0.02"-> -0.02    (siste tall i teksten brukes)
    "vet ikke"     -> None
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None if raw != raw else float(raw)
    matches = _NUMBER_RE.findall(str(raw))
    if not matches:
        return None
    s = matches[-1].strip().replace("−", "-")
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    percent = s.endswith("%")
    digits = re.sub(r"[^\d.]", "", s)
    if not digits or digits == ".":
        return None
    value = float(digits)
    if percent:
        value /= 100
    return -value if negative else value


def _valid(x):
    return x is not None and x == x  # ikke None og ikke NaN


def is_correct(pred, gold, rel_tol=REL_TOL):
    """Hovedregelen: relativ feil < rel_tol. Fasit 0 krever svar (nesten) 0."""
    if not _valid(pred):
        return False
    pred, gold = float(pred), float(gold)
    if gold == 0:
        return abs(pred) < 1e-9
    return abs(pred - gold) <= rel_tol * abs(gold)


def _round(value):
    step = Decimal(1).scaleb(-DECIMALS)  # 0.001
    return float(Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)) + 0.0


def format_answer(value):
    """Svaret avrundet til DECIMALS desimaler (for visning og tilleggsmålet)."""
    return _round(value) if _valid(value) else None


def is_correct_rounded(pred, gold):
    """Tilleggsmål: like etter avrunding til DECIMALS desimaler."""
    return _valid(pred) and _round(pred) == _round(gold)


def is_scale_error(pred, gold):
    """Riktig tall i feil skala: feil nå, men riktig delt eller ganget med 100."""
    if not _valid(pred) or is_correct(pred, gold):
        return False
    return is_correct(float(pred) / 100, gold) or is_correct(float(pred) * 100, gold)


def classify_failure(correct, answer, gold, gold_support, stopped=None):
    """Hvilket ledd feilet? Fast rekkefølge, det første som slår til gjelder. Felles for RAG-pipelinen og A.

    ingen        svaret er riktig
    maks_steg    (bare agenten) agenten nådde MAX_AGENT_STEPS uten å svare
    kalkulator   ikke noe tallsvar: programmet kunne ikke kjøres (RAG-pipelinen) / svaret kunne ikke leses (agenten)
    formatering  riktig tall, men feil skala (×100 eller ÷100)
    retrieval    ikke alle gold_inds ble hentet
    uttrykk      evidensen ble hentet, men regnestykket ble feil
    """
    if correct:
        return "ingen"
    if stopped == "maks_steg":
        return "maks_steg"
    if not _valid(answer):
        return "kalkulator"
    if is_scale_error(answer, gold):
        return "formatering"
    if not gold_support:
        return "retrieval"
    return "uttrykk"


if __name__ == "__main__":
    # Rask sjekk fra prosjektroten: python -m src.evaluering
    assert parse_answer("14%") == 0.14 and parse_answer("$1,234.5") == 1234.5
    assert parse_answer("(32)") == -32.0 and parse_answer("Answer: -0.02") == -0.02
    assert parse_answer("vet ikke") is None and parse_answer(0.5) == 0.5
    assert is_correct(0.017135, 0.01714) and not is_correct(0.0251, 0.01714)
    assert is_scale_error(1.7135, 0.01714) and not is_correct(1.9, -1.9)
    assert not is_correct(0.0002, 0.00001) and is_correct_rounded(0.0002, 0.00001)
    assert is_correct_rounded(1.2465000000000002, 1.2465) and format_answer(-0.0001) == 0.0
    assert not is_correct(None, 0.5) and not is_correct(float("nan"), 0.5)
    assert classify_failure(False, None, 1, True, stopped="maks_steg") == "maks_steg"
    assert classify_failure(False, None, 1, True) == "kalkulator"
    assert classify_failure(False, 100.0, 1.0, True) == "formatering"
    assert classify_failure(False, 2.0, 1.0, False) == "retrieval"
    assert classify_failure(False, 2.0, 1.0, True) == "uttrykk"
    print("evaluering.py: alle sjekker OK")
