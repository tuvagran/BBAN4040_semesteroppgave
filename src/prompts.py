"""Felles promptformuleringer for RAG-pipelinen og agenten.

Avtalt: agentens prompt gjenbruker formuleringene fra RAG-pipelinen. Reglene for tallformat og
kalkulatorprogram står derfor ÉN gang her, og begge promptene bygges av dem.
"""

# Tallformat (avtalt): ingen tusenskille, ingen valutategn, svar som ett tall i desimalform.
NUMBER_RULES = """- Write numbers without thousands separators and without currency signs: 1234.5, not $1,234.5
- Write negative numbers with a minus sign: -32, not (32)
- Percentages and ratios are answered as fractions in decimal form. Do NOT multiply by 100.
  A change from 50 to 60 is 0.2, not 20. 12.4% is written 0.124.
- Use only numbers that appear in the report."""

# Kalkulatorformatet (src/kalkulator.py, run_program).
PROGRAM_RULES = """- Steps are separated by ", ". Each step is op(a, b) where op is one of: add, subtract, multiply, divide.
- a and b are numbers, or #0, #1, ... meaning the result of step 0, 1, ...
- Never nest operations inside each other. Use #0, #1, ... instead:
  write add(60.1, 161.9), divide(161.9, #0), not divide(161.9, add(60.1, 161.9))."""

# --- RAG-pipelinen: én generering, ett program ---
B2_SYSTEM_PROMPT = f"""You answer numerical questions about financial reports.
You do NOT compute the answer yourself. Instead you write ONE calculator program that computes it.

Program format:
{PROGRAM_RULES}

Numbers:
{NUMBER_RULES}

Always write a program, even if you are unsure.
Reply with the program only, on one line. No explanation."""

# Oppdiktet eksempel (one-shot), ikke fra FinQA, så ingen fasit lekker inn.
B2_EXAMPLE_USER = """Context:
[table_2 | tabell]
net sales: 2019 = $ 1500; 2018 = $ 1200

Question: what was the percentage change in net sales from 2018 to 2019?"""
B2_EXAMPLE_ASSISTANT = "subtract(1500, 1200), divide(#0, 1200)"
