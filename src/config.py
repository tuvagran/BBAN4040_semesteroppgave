"""Felles innstillinger for RAG-pipelinen og agenten.

Alt som kan påvirke resultatet og som skal være likt i begge betingelsene, står her.
Begge notebookene importerer fra denne filen:
    import sys
    sys.path.insert(0, "..")          # prosjektroten, når notebooken ligger i notebooks/
    from src.config import *

Den eneste forskjellen mellom RAG-pipelinen og agenten skal være hvem som styrer søk og regning.
"""
from pathlib import Path

# --- Stier ---
ROOT = Path(__file__).resolve().parent.parent      # prosjektroten (BBAN4040/)
DATA_RAW = ROOT / "data" / "raw"                   # hele FinQA, lastes ned med scripts/last_ned_finqa.py
DATA_DIR = ROOT / "data"
FINQA_UTVALG = DATA_DIR / "finqa_utvalg"           # bare rapportsidene utvalgene trenger (ligger i repoet)
RESULTS_DIR = ROOT / "results"

# --- Språkmodell ---
# gpt-4.1 (fullversjonen, ikke mini). Støtter verktøykall og, i motsetning til
# resonneringsmodellene (gpt-5/o-serien), temperature-parameteren.
# Byttet fra gpt-5.6-luna 2026-10-23 for å kunne sette temperatur 0.
# Testet 2026-10-23: aliaset peker på snapshot gpt-4.1-2025-04-14, temperature=0 godtas.
MODEL = "gpt-4.1"
# 0 = mest mulig deterministisk: samme spørsmål gir (nesten alltid) samme svar ved ny kjøring.
# Likt for RAG-pipelinen og agenten. None = parameteren sendes ikke.
TEMPERATURE = 0

# --- Utvalg ---
SEED = 42
N_DEV = 50              # dev-utvalg: promptene utvikles her
N_VAL = 100             # valideringsutvalg fra train.json: designvalgene testes her (pilot_val.ipynb)
N_TEST = 200            # testutvalg: kjøres ÉN gang, når begge systemene er ferdige
UTVALG_DEV = DATA_DIR / "utvalg_dev.csv"
UTVALG_TEST = DATA_DIR / "utvalg_test.csv"      # fast utvalg, trekkes ikke på nytt (se src/utvalg.py)
UTVALG_VAL = DATA_DIR / "utvalg_val.csv"
ALLOWED_OPS = ("add", "subtract", "multiply", "divide")   # filter på qa.program

# --- Korpus og søk (felles search for RAG-pipelinen og agenten) ---
RETRIEVAL_SCOPE = "dokument"     # søk bare i spørsmålets egen rapportside
CHUNK_STRATEGY = "setning"       # én FinQA-setning per chunk, én tabellrad per chunk
SENTENCES_PER_CHUNK = 3          # brukes bare når CHUNK_STRATEGY = "gruppe"
INDEX_BACKEND = "embedding"      # "embedding" (dense) eller "tfidf" (sparse); valgt på valideringsutvalget
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5                        # antall chunks per søk, likt for RAG-pipelinen og hvert agentsøk

# --- Agent ---
MAX_AGENT_STEPS = 8              # maks antall modellkall per spørsmål for agenten

# --- Demo ---
# Antall testspørsmål som kjøres live i demoen (de første i data/utvalg_test.csv).
# Påvirker ikke resultatene: demoen lagrer ingenting og står ikke i settings().
N_DEMO = 8
# Pris for gpt-4.1 i dollar per million tokens (OpenAI, standard API). Brukes bare til å
# vise et kostnadsestimat for demoen.
PRIS_INPUT_PER_MTOK = 2.00
PRIS_OUTPUT_PER_MTOK = 8.00

# --- Evaluering ---
REL_TOL = 0.01                   # hovedregel: relativ feil < 1 % mot qa.exe_ans
DECIMALS = 3                     # tilleggsmål: likhet etter avrunding til 3 desimaler


def settings() -> dict:
    """Innstillingene som logges sammen med hvert resultat."""
    return {
        "model": MODEL, "temperature": TEMPERATURE, "seed": SEED,
        "retrieval_scope": RETRIEVAL_SCOPE, "chunk_strategy": CHUNK_STRATEGY,
        "index_backend": INDEX_BACKEND, "embedding_model": EMBEDDING_MODEL, "top_k": TOP_K,
        "max_agent_steps": MAX_AGENT_STEPS, "rel_tol": REL_TOL,
    }
