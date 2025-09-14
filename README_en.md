# AI BCC: Smart Push Personalization (Case 1)

This repo builds a smart push-automation for a bank: we analyze 3 months of behavior, compute expected customer benefit per product, choose the best product, and generate personalized, on-brand push texts.

## Pipeline

- Canonicalize & Features
  - Inputs: `case_1/clients.csv`, `case_1/client_*_transactions_3m.csv`, `case_1/client_*_transfers_3m.csv`
  - Outputs:
    - `data/canonicalized/client_{i}_transactions_3m_canon.csv`
    - `data/features_per_client.csv`

- Benefits & Top-4 (with mutual-exclusion + debug)
  - Computes incremental product benefits (vs debit baseline), caps, and realization (adoption) factors.
  - Adds mutual-exclusion: when a deposit is clearly the behavior, credit eligible spend is shrunk.
  - Outputs:
    - `data/benefits_per_client.csv`
    - `data/top4_per_client.csv`
    - `data/benefits_debug_per_client.csv` (why the choice: realizations, caps, eligible spend before/after MX)

- LLM Copy Engine (Gemini 2.5 Flash)
  - Builds a structured context per client and calls Gemini with retries, schema-enforced JSON, sanitization, and auto-repair (CTA / "вы" / length 180–220).
  - Outputs:
    - `data/personalized_pushes.csv`
    - `data/llm_copy_debug.jsonl` (every attempt: errors, sanitizer, validator reasons)

- Export
  - Joins the first three scripts, applies send / no-send policy, validates, and emits the final 3-column CSV.
  - Outputs:
    - `data/submissions/pushes_final.csv` (exact: `client_code,product,push_notification`)
    - `data/submissions/pushes_with_debug.csv` (richer explainer)

## Quickstart

### 0) Python deps

```bash
pip install -r requirements.txt
```

### 1) Environment
Create .env at repo root:
```bash
GOOGLE_API_KEY=AIza...your_key...
```

### 2) Run everything
Bash one-liner:
```bash
bash run_all.sh
```

### 3) Check outputs
- Submission: data/submissions/pushes_final.csv
- Debug: data/submissions/pushes_with_debug.csv
- LLM debug: data/llm_copy_debug.jsonl
- Benefits debug: data/benefits_debug_per_client.csv

## How build_features.py decides

- Credit card: 10% on (Top-3 ∪ Online) incremental to baseline 1%, with monthly cap and realization from a data-driven eligibility score.
- Premium card: base 2–4% (by balance) + 4% on boosted categories, minus baseline, plus ATM/transfer savings, times a realization factor.
- Travel card: 4% minus baseline on travel/taxi spend, times realization based on taxi/hotel activity.
- Deposits: pick type (Saving/Accumulation/Multi) by behavior, then multiply by realization.
- FX: % savings on FX volume, with higher realization when FX volume is sizable
- Mutual-exclusion: when deposit behavior dominates (high balance, low spend, or deposit benefit near credit), shrink credit’s eligible spend (10–60% diversion).
- Tie-break: stable ordering if benefits are equal within epsilon.

## LLM Safety & Quality
- Structured JSON with enum CTA: the model must return {"message": "...", "cta": "<one-of>"}.
- Strict sanitizer handles code fences, arrays, stray text.
- Validator checks: length 180–220, exactly one CTA (whitelist), "вы/вас", <=1 emoji, <=1 "!".
- Auto-repair: if short, insert one factual clause before CTA; normalize conjugated CTA; enforce "вы"; shrink if long.
- Retries: exponential backoff; on JSON/schema/validation failures, we echo the exact errors back to the model.

## Config knobs

Edit these in build_benefits.py, generate_pushes.py:

build_benefits.py:
```bash
BASELINE_CB, CREDIT_CAP_MONTHLY, TRAVEL_REALIZATION_BASE, PREMIUM_REALIZATION_*, etc.
Mutual-exclusion thresholds in benefits_for_client_row().
```

generate_pushes.py:
```bash
MIN_LEN, MAX_LEN, ALLOWED_CTA
LLM_MAX_RETRIES, TOKEN_BUDGET
Tone rules: REQUIRE_VY_LOWER, emoji/exclamation caps
```

### Deliverables

data/submissions/pushes_final.csv - final 3-column CSV
data/submissions/pushes_with_debug.csv - explainable audit trail
