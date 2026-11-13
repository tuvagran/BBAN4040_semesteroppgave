# BBAN4040 Språkmodeller – semesteroppgave (gruppe 3)

**Problemstilling:** I hvilken grad gir en mer ressurskrevende LLM-agent flere riktige svar på numeriske spørsmål i FinQA enn en fast RAG-pipeline med lavere ressursbehov, når begge systemene har tilgang til den samme kalkulatoren?

To systemer besvarer de samme spørsmålene fra FinQA med samme språkmodell, søk og kalkulator. Det eneste som skiller dem, er hvem som styrer verktøybruken:

| | Søk | Regning | Hvem styrer |
|---|---|---|---|
| **RAG-pipeline**: fast kjede | Ett søk, med spørsmålet som søketekst | Ett helt program, kjørt av kalkulatoren | Programkoden |
| **Agent** | Ett eller flere søk, med egne søketekster | Ett regnesteg om gangen | Språkmodellen (maks 8 modellkall) |

## Felles oppsett

Alt som skal være likt i begge systemene, ligger i `src/` og står i `src/config.py`:

- **Språkmodell:** `gpt-4.1` (snapshot `gpt-4.1-2025-04-14`) via OpenAI Responses API, temperatur 0
- **Korpus:** én chunk per setning og per tabellrad, søk bare i spørsmålets egen rapportside
- **Søk:** dense retrieval med `all-MiniLM-L6-v2`, cosinuslikhet, fem treff per søk
- **Kalkulator:** addisjon, subtraksjon, multiplikasjon og divisjon (`src/kalkulator.py`)
- **Korrekthet:** relativ feil under 1 % mot FinQAs `exe_ans`. Tilleggsmål: likhet etter avrunding til tre desimaler.

## Kjøre notebookene (ca. 1 minutt, ingen API-kall)

```bash
pip install -r requirements.txt
```

Kjør alle cellene i `notebooks/RAG-pipeline.ipynb`, `notebooks/Agent.ipynb` og `notebooks/pilot_val.ipynb`. Notebookene gjør **ingen API-kall** og trenger ingen API-nøkkel. De viser hvert steg i metoden og leser kjøringene i `results/`. Det er disse kjøringene resultatene i rapporten kommer fra.

| Notebook | Tid | Leser |
|---|---|---|
| `RAG-pipeline.ipynb` | ca. 20 s | `results/rag_pipeline_test.jsonl` (200 testspørsmål) |
| `Agent.ipynb` | ca. 20 s | `results/agent_test.jsonl` (200 testspørsmål) |
| `pilot_val.ipynb` | ca. 25 s | `results/rag_pipeline_val_designvalg.jsonl` (6 × 100 valideringsspørsmål) |

Tidene er målt med embeddingmodellen allerede lastet ned. Første gang kommer nedlastingen fra Hugging Face i tillegg (ca. 90 MB, vanligvis under ett minutt).

## Ekstra: live-demo mot API-et (valgfritt)

Som et supplement kan RAG-pipelinen og agenten kjøres **live mot OpenAI** på de 8 første testspørsmålene. Demoen ligger som et eget steg **sist** i `RAG-pipeline.ipynb` (seksjon 12) og `Agent.ipynb` (seksjon 9), etter at de lagrede resultatene er vist. Den er av som standard.

1. Legg inn en API-nøkkel: `cp .env.example .env` og fyll inn `OPENAI_API_KEY`.
2. Sett `LIVE_DEMO = True` i den siste kodecellen i notebooken, og kjør cellen.

| Notebook | Tid | Kostnad (målt) |
|---|---|---|
| `RAG-pipeline.ipynb` | ca. 15 s | ca. 1 cent |
| `Agent.ipynb` | ca. 50 s | 6–9 cent |

Demoen bruker de samme funksjonene som resten av notebooken. Live-svarene vises ved siden av svarene fra den lagrede testkjøringen. Med temperatur 0 blir de stort sett like, men små forskjeller kan forekomme. Demoen lagrer ingenting og endrer ikke resultatene i `results/`. Antall spørsmål kan endres med `N_DEMO` i `src/config.py`.

**Kjøre hele utvalget på nytt** (ikke nødvendig for å vurdere oppgaven): `RERUN = True` i kjørecellen kjører alle spørsmålene på nytt og **overskriver** `results/`. Det tar ca. 4 min for RAG-pipelinen, 20 min for agenten og 11 min for piloten. Agenten alene koster da ca. 2 dollar.

## Data og utvalg

FinQA (Chen et al., 2021, MIT-lisens, se `data/raw/LICENSE_FinQA`). Bare spørsmål der fasitprogrammet bruker de fire regneartene er med.

| Utvalg | Spørsmål | Rapportsider (i repoet) | Kilde | Antall | Brukes til |
|---|---|---|---|---|---|
| dev | `data/utvalg_dev.csv` | `data/finqa_utvalg/dev.json` | `dev.json` | 50 | Utvikle promptene |
| val | `data/utvalg_val.csv` | `data/finqa_utvalg/val.json` | `train.json` | 100 | Teste designvalgene (`notebooks/pilot_val.ipynb`) |
| test | `data/utvalg_test.csv` | `data/finqa_utvalg/test.json` | `test.json` | 200 | Endelig sammenligning, kjørt én gang. De 8 første brukes i live-demoen. |

**Hele datasettet ligger ikke i repoet.** `data/finqa_utvalg/` inneholder bare de 350 rapportsidene som utvalgene trenger (2,7 MB), uendret fra FinQA. Det er nok til å kjøre alle tre notebookene. Hele FinQA (ca. 100 MB) trengs bare for å trekke dev og val på nytt og for å teste kalkulatoren på hele dev.json. Det lastes ned til `data/raw/` med:

```bash
python scripts/last_ned_finqa.py            # dev.json, test.json og train.json
python -m src.utvalg                        # trekker dev og val på nytt (blir identiske, seed 42)
python scripts/valider_kalkulator.py        # tester kalkulatoren mot alle fasitprogrammene i dev.json
```

Testutvalget er fast og trekkes ikke på nytt. Det ble trukket med seed 42 blant de 985 testspørsmålene som ikke står i `data/ekskludert_fra_test.csv`. Fila inneholder 100 testspørsmål fra en tidlig utprøving av RAG-pipelinen, som derfor ble holdt utenfor.

## Struktur

```
src/
  config.py        alle innstillinger, felles for RAG-pipelinen og agenten
  utvalg.py        utvalgene (dev, val, test)
  korpus.py        chunking, indeks og søk
  kalkulator.py    kalkulatoren
  prompts.py       felles tallregler og RAG-pipeline-prompten
  llm.py           modellkall, tokentelling og kostnadsestimat
  evaluering.py    korrekthetsregler og feilsteder
  logg.py          loggformat (én JSON-linje per spørsmål)
  rapport.py       evalueringstabellene og sammenligningen i demoen
notebooks/
  RAG-pipeline.ipynb        RAG-pipeline: fast kjede med kalkulator
  Agent.ipynb               Agent: modellen styrer søk og kalkulator
  pilot_val.ipynb           designvalgene testet på valideringsutvalget
scripts/
  last_ned_finqa.py         laster ned hele FinQA til data/raw/
  valider_kalkulator.py     tester kalkulatoren mot fasitprogrammene i dev.json
results/
  rag_pipeline_test.jsonl, agent_test.jsonl   testresultatene, én linje per spørsmål
  rag_pipeline_val_designvalg.jsonl           resultatene fra pilot_val.ipynb
data/
  finqa_utvalg/             rapportsidene til spørsmålene i utvalgene
  raw/                      lisens (hele FinQA lastes ned hit)
  utvalg_*.csv              utvalgene
  ekskludert_fra_test.csv   100 testspørsmål holdt utenfor testutvalget
```

Notebookene kjøres fra `notebooks/`. Utvalget velges med `SPLIT` i første kodecelle (`"test"` er satt nå). Lagrede resultater i `results/` brukes bare så lenge oppsettet i `src/config.py` er uendret. Embeddingmodellen kan brukes uten nettsjekk etter første nedlasting med `HF_HUB_OFFLINE=1`.
