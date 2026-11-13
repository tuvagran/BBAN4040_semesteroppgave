"""Felles evaluering og feilanalyse for RAG-pipelinen og agenten.

Begge notebookene lager tabellene med de samme funksjonene, så målene er regnet ut likt:
    from src.rapport import oppsummering, failure_log
    tabeller = oppsummering(resultater)     # dict med DataFrames
    logg = failure_log(resultater)          # én rad per feil svar

`resultater` er en DataFrame med de avtalte loggfeltene (src/logg.py), der API-feil
(`error` satt) allerede er fjernet.

Den parvise sammenligningen av to systemer på de samme spørsmålene (McNemar):
    mcnemar_sammenligning(agent, rag, kolonne="correct", navn_a="agenten", navn_b="RAG-pipelinen")

Live-demoen bruker demo_sammenligning:
    sammendrag, tabell = demo_sammenligning(demo, RESULTS_DIR / "agent_test.jsonl")
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest     # eksakt McNemar-test: binomialtest på de uenige parene

from src.evaluering import is_correct
from src.llm import kostnad_dollar
from src.logg import read_jsonl

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Forslag til neste endring per feilsted (som next_change i øvingen)
NEXT_CHANGE = {
    "retrieval":   "bedre chunking (segment/enhet i tabellrader), høyere top_k eller flere søk",
    "uttrykk":     "tydeligere prompt: velg riktig rad/år, fortegn og nevner",
    "kalkulator":  "prompt: krev alltid et program/tallsvar i riktig format",
    "formatering": "prompt: presiser andel vs. prosent (merk: FinQA-fasiten er selv inkonsekvent)",
    "maks_steg":   "færre, mer presise søk eller høyere stegsgrense",
}


def andel_med_ki(treff: pd.Series, z: float = 1.96) -> str:
    """Andel med 95 % Wilson-konfidensintervall, f.eks. '59% [49%, 68%]'."""
    treff = treff.astype(bool)
    n, p = len(treff), treff.mean()
    if n == 0:
        return "–"
    midt = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return f"{p:.0%} [{midt - margin:.0%}, {midt + margin:.0%}]"


def oppsummering(r: pd.DataFrame) -> dict:
    """Alle evalueringstabellene for én betingelse."""
    r = r.copy()
    r["regnesteg"] = np.where(r["n_steps"] == 1, "1 steg", "2+ steg")
    t = {}

    # 1. Nøyaktighet (hovedregel og tilleggsmål)
    t["noyaktighet"] = pd.DataFrame([
        {"mål": "riktig, relativ feil < 1 % (hovedregel)", "andel [95 % KI]": andel_med_ki(r["correct"]),
         "antall": int(r["correct"].sum())},
        {"mål": "riktig, 3 desimaler (tilleggsmål)", "andel [95 % KI]": andel_med_ki(r["correct_rounded"]),
         "antall": int(r["correct_rounded"].sum())},
    ])
    # 2. Retrieval recall mot gold_inds
    t["recall"] = pd.DataFrame([
        {"mål": "alle gold_inds hentet", "andel [95 % KI]": andel_med_ki(r["gold_support_retrieved"])},
        {"mål": "minst én gold_ind hentet", "andel [95 % KI]": andel_med_ki(r["any_gold_retrieved"])},
    ])
    # 3. Nøyaktighet med og uten evidens
    e = r.groupby("gold_support_retrieved")["correct"].agg(antall="size", riktige="sum", andel="mean")
    e.index = e.index.map({True: "alle gold hentet", False: "gold mangler"})
    e["andel"] = e["andel"].map("{:.0%}".format)
    t["etter_evidens"] = e
    # 4. Ett vs. flere regnesteg (avtalt inndeling)
    t["etter_steg"] = r.groupby("regnesteg").agg(
        antall=("correct", "size"), riktige=("correct", "sum"), noyaktighet=("correct", andel_med_ki),
        alle_gold_hentet=("gold_support_retrieved", lambda s: f"{s.mean():.0%}"))
    # 5. Feilsteder
    f = r["failure_stage"].value_counts().rename("antall").to_frame()
    f["andel"] = (f["antall"] / len(r)).map("{:.0%}".format)
    t["feilsted"] = f
    # 6. Kostnad per spørsmål
    t["kostnad"] = pd.DataFrame([{
        "modellkall (snitt)": round(float(r["n_model_calls"].mean()), 2),
        "verktøykall (snitt)": round(float(r["n_tool_calls"].mean()), 2),
        "input_tokens (snitt)": round(float(r["input_tokens"].mean())),
        "output_tokens (snitt)": round(float(r["output_tokens"].mean())),
        "reasoning_tokens (snitt)": round(float(r["reasoning_tokens"].mean())),
        "sekunder (snitt)": round(float(r["seconds"].mean()), 1),
        "tokens totalt": int(r["input_tokens"].sum() + r["output_tokens"].sum()),
    }]).T.rename(columns={0: r["condition"].iloc[0]})
    return t


def _program_numbers(program: str) -> list:
    """Tallene i et FinQA-program, uten #0 (mellomresultat) og const_100 (konstanter)."""
    args = [a.strip() for step in re.findall(r"\(([^)]*)\)", program) for a in step.split(",")]
    return [float(NUM_RE.search(a.replace("%", "")).group()) for a in args
            if a and not a.startswith(("#", "const_")) and NUM_RE.search(a)]


def gold_numbers_in_context(row) -> bool:
    """Uavhengig evidenstest: står alle tallene fra fasitprogrammet i det systemet hentet?"""
    text = " ".join(c["text"] for c in row["retrieved_chunks"] or []).replace(",", "")
    found = {abs(float(x)) for x in NUM_RE.findall(text)}
    return all(any(abs(abs(n) - f) < 1e-9 for f in found) for n in _program_numbers(row["gold_program"]))


def _answer_subtype(raw) -> str:
    """Hva slags svar når det ikke ble noe tall (failure_stage = kalkulator)."""
    raw = raw or ""
    if re.search(r"(add|subtract|multiply|divide)\([^)]*(add|subtract|multiply|divide)\(", raw):
        return "nøstet program"
    if re.fullmatch(r"\s*-?[\d.,]+%?\s*", raw):
        return "bare et tall"
    if not re.search(r"\d", raw):
        return "avstod / tekst"
    return "annet format"


def failure_log(r: pd.DataFrame) -> pd.DataFrame:
    """Én rad per feil svar, med feilsted, uavhengig evidenstest og neste endring."""
    feil = r[~r["correct"].astype(bool)].copy()
    feil["tall_i_kontekst"] = feil.apply(gold_numbers_in_context, axis=1)
    feil["undertype"] = np.where(feil["failure_stage"] == "kalkulator", feil["raw_answer"].map(_answer_subtype), "")
    feil["next_change"] = feil["failure_stage"].map(NEXT_CHANGE)
    cols = ["id", "question", "gold_answer", "answer", "raw_answer", "gold_program", "gold_inds",
            "tall_i_kontekst", "failure_stage", "undertype", "next_change"]
    return feil[cols].sort_values("failure_stage")


def failure_oversikt(logg: pd.DataFrame, n_total: int) -> pd.DataFrame:
    """Feilsteder, og hvor ofte fasittallene faktisk stod i det systemet hentet."""
    o = logg.groupby("failure_stage").agg(antall=("id", "size"), fasittall_i_kontekst=("tall_i_kontekst", "sum"))
    o["andel_av_alle_spm"] = (o["antall"] / n_total).map("{:.0%}".format)
    return o.sort_values("antall", ascending=False)


def mcnemar_sammenligning(a: pd.DataFrame, b: pd.DataFrame, kolonne: str = "correct",
                          navn_a: str = "A", navn_b: str = "B") -> dict:
    """Parvis sammenligning av to systemer på de samme spørsmålene, med eksakt McNemar-test.

    Radene kobles på id. Bare de uenige spørsmålene (der bare det ene systemet har rett)
    sier noe om hvilket som er best. Er systemene like gode, er hvert uenige spørsmål et
    myntkast, så testen er en tosidig binomialtest med p = 0.5, som i pilot_val.ipynb.
    `kolonne`: "correct" (hovedregelen, 1 %) eller "correct_rounded" (3 desimaler).
    """
    par = a[["id", kolonne]].merge(b[["id", kolonne]], on="id", suffixes=("_a", "_b"), validate="one_to_one")
    if len(par) != len(a) or len(par) != len(b):
        raise ValueError(f"Filene har ikke de samme spørsmålene ({len(a)}, {len(b)}, {len(par)} felles)")
    riktig_a = par[f"{kolonne}_a"].fillna(False).astype(bool)
    riktig_b = par[f"{kolonne}_b"].fillna(False).astype(bool)
    bare_a = int((riktig_a & ~riktig_b).sum())
    bare_b = int((~riktig_a & riktig_b).sum())
    n_uenige = bare_a + bare_b
    return {
        "regel": kolonne,
        "spørsmål": len(par),
        f"riktige ({navn_a})": int(riktig_a.sum()),
        f"riktige ({navn_b})": int(riktig_b.sum()),
        "begge riktig": int((riktig_a & riktig_b).sum()),
        "begge feil": int((~riktig_a & ~riktig_b).sum()),
        f"bare {navn_a} riktig": bare_a,
        f"bare {navn_b} riktig": bare_b,
        # Ingen uenige spørsmål: ingen forskjell å teste, p settes til 1 (som i piloten).
        "p (McNemar, eksakt)": float(binomtest(bare_b, n_uenige, 0.5).pvalue) if n_uenige else 1.0,
    }


def demo_sammenligning(demo: pd.DataFrame, lagret_fil) -> tuple[dict, pd.DataFrame]:
    """Live-svarene fra demoen ved siden av svarene fra den lagrede testkjøringen.

    Sammendraget viser antall riktige, tid, tokens og estimert kostnad i dollar for demoen.
    Med temperatur 0 bør live-svarene stort sett være like de lagrede, men små forskjeller
    kan forekomme (modellen er ikke helt deterministisk).
    """
    tabell = demo.copy()
    sammendrag = {"spørsmål": len(tabell), "riktige (live)": int(tabell["correct"].fillna(False).sum())}
    if lagret_fil.exists():
        lagret = read_jsonl(lagret_fil).set_index("id")
        tabell["svar_testkjøring"] = tabell["id"].map(lagret["answer"])
        tabell["riktig_testkjøring"] = tabell["id"].map(lagret["correct"])
        # Samme svar som i testkjøringen (innen 1 %, som hovedregelen)?
        tabell["samme_svar"] = [is_correct(live, lagret_svar) if pd.notna(lagret_svar) else pd.isna(live)
                                for live, lagret_svar in zip(tabell["answer"], tabell["svar_testkjøring"])]
        sammendrag["riktige (lagret testkjøring)"] = int(tabell["riktig_testkjøring"].fillna(False).sum())
        sammendrag["samme svar som testkjøringen"] = int(tabell["samme_svar"].sum())
    inn, ut = tabell["input_tokens"].fillna(0).sum(), tabell["output_tokens"].fillna(0).sum()
    sammendrag.update({"api_feil": int(tabell["error"].notna().sum()),
                       "sekunder": round(float(tabell["seconds"].sum()), 1),
                       "tokens": int(inn + ut),
                       "kostnad i dollar (estimat)": round(float(kostnad_dollar(inn, ut)), 3)})
    return sammendrag, tabell
