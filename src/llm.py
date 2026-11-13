"""Felles kall til språkmodellen for RAG-pipelinen og agenten.

Begge bruker samme klient, modell og innstillinger (src/config.py), og samme API
(OpenAI Responses API, som i forelesning 6). Tokens telles likt i begge.

    from src.llm import call_model, usage
    response = call_model(input=[{"role": "user", "content": "..."}], instructions="...")
    response.output_text, usage(response)
"""
import os

from src.config import (MODEL, PRIS_INPUT_PER_MTOK, PRIS_OUTPUT_PER_MTOK, ROOT,
                        TEMPERATURE)

_client = None


def _find_key() -> str | None:
    """OPENAI_API_KEY fra .env i prosjektroten, ellers fra miljøvariabelen. None hvis ingen.

    Nøkkelen skrives aldri ut.
    """
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    return os.environ.get("OPENAI_API_KEY") or None


def har_api_nokkel() -> bool:
    """True hvis en API-nøkkel er satt. Notebookene hopper over live-cellene ellers."""
    return _find_key() is not None


def _load_key() -> str:
    key = _find_key()
    if key is None:
        raise RuntimeError("Fant ingen OPENAI_API_KEY i .env eller som miljøvariabel")
    return key


def kostnad_dollar(input_tokens: float, output_tokens: float) -> float:
    """Estimert pris i dollar for et antall tokens (prisene står i src/config.py).

    Resonneringstokens telles allerede med i output_tokens.
    """
    return input_tokens / 1e6 * PRIS_INPUT_PER_MTOK + output_tokens / 1e6 * PRIS_OUTPUT_PER_MTOK


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        # timeout: et kall som henger, avbrytes etter 60 s og prøves på nytt (max_retries).
        # Uten dette venter klienten opptil 10 minutter per forsøk.
        _client = OpenAI(api_key=_load_key(), timeout=60, max_retries=3)
    return _client


def call_model(input, instructions: str = None, tools: list = None, tool_choice: str = None):
    """Ett modellkall. `input` er en liste med meldinger/elementer (Responses API).

    tool_choice="none": verktøyene er kjent for modellen, men den kan ikke kalle dem
    (brukes til agentens tvungne sluttsvar på siste steg).
    """
    kwargs = {"model": MODEL, "input": input}
    if instructions:
        kwargs["instructions"] = instructions
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    if TEMPERATURE is not None:          # resonneringsmodeller (gpt-5/o) godtar ikke temperature
        kwargs["temperature"] = TEMPERATURE
    return get_client().responses.create(**kwargs)


def usage(response) -> dict:
    """Tokenbruk for ett kall. reasoning_tokens er skjulte «tenke-tokens» (faktureres som output)."""
    u = response.usage
    details = getattr(u, "output_tokens_details", None)
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
    }
