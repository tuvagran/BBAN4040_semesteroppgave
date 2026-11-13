"""Felles logg for RAG-pipelinen og agenten: én JSON-linje per spørsmål, samme felt i begge.

    from src.logg import make_record, write_jsonl, read_jsonl, same_settings
    record = make_record(condition="RAG-pipeline", id=..., question=..., ...)
    write_jsonl([record, ...], RESULTS_DIR / "rag_pipeline_dev.jsonl")
    df = read_jsonl(RESULTS_DIR / "rag_pipeline_dev.jsonl")

Feltene er avtalt. Felt som ikke er relevante for en betingelse, settes til None.
"""
import json

import pandas as pd

LOG_FIELDS = [
    # Hvilket spørsmål og hvilken betingelse
    "condition",            # "RAG-pipeline" eller "Agent"
    "split",                # "dev" eller "test"
    "id", "question",
    "gold_answer",          # qa.exe_ans
    "gold_program", "gold_inds", "n_steps",
    # Svaret
    "answer",               # parset tall (float) eller None
    "raw_answer",           # råteksten fra modellen
    # Hva systemet gjorde
    "tool_calls",           # liste: {"name", "arguments", "output"} (RAG-pipelinen: run_program, agenten: search/calculator_tool)
    "retrieved_chunks",     # liste: {"chunk_id", "source_ids", "text"} for alle hentede chunks
    "n_model_calls", "n_tool_calls",
    "stopped",              # hvorfor kjøringen stoppet: "svar", "tvunget_svar" (agenten, siste steg) eller "maks_steg"
    "input_tokens", "output_tokens", "reasoning_tokens",
    "seconds",
    "error",                # API-feil eller lignende (ikke feil svar)
    # Evaluering (src/evaluering.py)
    "correct",              # hovedregel: relativ feil < 1 %
    "correct_rounded",      # tilleggsmål: 3 desimaler
    "gold_support_retrieved", "any_gold_retrieved",
    "failure_stage",        # ingen / retrieval / uttrykk / kalkulator / formatering / (agenten: maks_steg)
    # Oppsett
    "model", "settings",
]


def make_record(**fields) -> dict:
    """Én loggrad med alle LOG_FIELDS. Ukjente felt gir feil, manglende blir None."""
    unknown = set(fields) - set(LOG_FIELDS)
    if unknown:
        raise KeyError(f"Ukjente loggfelt: {sorted(unknown)}")
    return {f: fields.get(f) for f in LOG_FIELDS}


def write_jsonl(records: list, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def same_settings(df: pd.DataFrame, current: dict, ignore: tuple = ()) -> bool:
    """True hvis alle radene i en lagret kjøring ble laget med innstillingene `current`.

    Brukes før en lagret resultatfil gjenbrukes: er modell, temperatur, top_k e.l. endret
    i src/config.py, er filen utdatert og må kjøres på nytt.
    `ignore`: felt som får variere mellom radene (f.eks. "top_k" i top_k-eksperimentet).
    """
    want = {k: v for k, v in current.items() if k not in ignore}
    return all({k: v for k, v in s.items() if k not in ignore} == want for s in df["settings"])
