"""Felles kalkulator for RAG-pipelinen og agenten.

Begge betingelsene skal importere herfra, slik at regningen er identisk:
- Agenten bruker `calculator_tool(op, a, b)` som verktøy, ett steg om
  gangen. Den kaller `calculate`, men kaster aldri feil: feil returneres som
  tekst ("FEIL: ...") slik at agenten kan lese den og prøve igjen.
  Hvert kall logges (se `get_log` og `reset_log`).
- RAG-pipelinen bruker `run_program(program)` på et helt FinQA-program,
  f.eks. "subtract(5829, 5735), divide(#0, 5735)".

Import fra en notebook i notebooks/-mappen:
    import sys
    sys.path.append("..")
    from src.kalkulator import run_program, calculate

Kalkulatoren runder ikke av og ganger ikke med 100. Avrunding og
sammenligning med fasit er evalueringens jobb.
"""
import re
from datetime import datetime

OPS = {
    "add":      lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide":   lambda a, b: a / b,
}

STEP_RE = re.compile(r"([a-z_]+)\(([^)]*)\)")


def clean_number(x):
    """Gjør et tall eller en tekst om til float. Felles for agenten og RAG-pipelinen.

    5           -> 5.0 (int/float returneres direkte som float)
    ' 1,234 '   -> 1234.0 (fjerner mellomrom og tusenskille-komma)
    '$500'      -> 500.0
    '5%'        -> 0.05 (FinQA tolker prosent som andel)
    Kaster ValueError hvis teksten ikke kan leses som et tall.
    """
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("$", "").replace(",", "")
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100
        return float(s)
    except ValueError:
        raise ValueError(f"Kan ikke lese tallet: {x!r}") from None


def calculate(op, a, b):
    """Ett regnesteg. Kaster feil ved ugyldig input (brukes av RAG-pipelinen)."""
    if op not in OPS:
        raise ValueError(f"Ukjent operasjon: {op}")
    return OPS[op](clean_number(a), clean_number(b))


# Logg over alle kall til calculator_tool. Nullstill mellom spørsmål.
TOOL_LOG = []


def calculator_tool(op, a, b):
    """Agentens verktøy. Som `calculate`, men kaster aldri feil.

    Returnerer et tall ved suksess, ellers en tekst "FEIL: <beskrivelse>".
    Hvert kall legges i TOOL_LOG (operasjon, a, b, resultat/feil, tidsstempel).
    """
    entry = {"op": op, "a": a, "b": b, "result": None, "error": None,
             "timestamp": datetime.now().isoformat()}
    try:
        result = calculate(op, a, b)
        entry["result"] = result
    except ZeroDivisionError:
        result = f"FEIL: deling på null ({op}({a}, {b}))"
        entry["error"] = result
    except ValueError as e:
        # Ukjent operasjon eller tall som ikke kan leses
        result = f"FEIL: {e}"
        entry["error"] = result
    except Exception as e:
        result = f"FEIL: {type(e).__name__}: {e}"
        entry["error"] = result
    TOOL_LOG.append(entry)
    return result


def get_log():
    """Returnerer en kopi av verktøyloggen."""
    return list(TOOL_LOG)


def reset_log():
    """Tømmer verktøyloggen. Kall denne før hvert nytt spørsmål."""
    TOOL_LOG.clear()


def parse_arg(arg, results):
    """Gjør ett argument om til et tall.

    '#0'        -> resultatet fra steg 0
    'const_100' -> 100.0 (FinQA-konstant), 'const_m1' -> -1.0
    '5829'      -> 5829.0 (ryddes med clean_number, '5%' -> 0.05)
    """
    arg = arg.strip()
    if arg.startswith("#"):
        return results[int(arg[1:])]
    if arg.startswith("const_"):
        c = arg[len("const_"):]
        return -1.0 if c == "m1" else float(c)
    return clean_number(arg)


def run_program(program):
    """Et helt program. Brukes i RAG-pipelinen. Returnerer resultatet av siste steg.

    Kaster feil (ValueError, ZeroDivisionError, IndexError) hvis programmet
    er ugyldig. Det er med vilje: ugyldige programmer skal telles som feil
    svar og logges, ikke gi et tilfeldig tall.
    """
    steps = STEP_RE.findall(program)
    if not steps:
        raise ValueError(f"Fant ingen steg i: {program!r}")
    results = []
    for op, args in steps:
        a, b = [parse_arg(x, results) for x in args.split(",")]
        results.append(calculate(op, a, b))
    return results[-1]


if __name__ == "__main__":
    # Rask sjekk: python src/kalkulator.py
    print(run_program("subtract(5829, 5735), divide(#0, 5735)"))  # ca. 0.0164
    print(calculate("multiply", 2, 3))                            # 6.0
