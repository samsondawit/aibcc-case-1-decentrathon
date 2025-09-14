"""
Data loading + category discovery + canonicalization + features
- Discovers raw categories from all client_*_transactions_3m.csv files
- Maps to canonical categories via exact -> fuzzy (RapidFuzz) -> Gemini 2.5 Flash fallback
- Caches LLM mappings (taxonomy_cache.json) to avoid re-calling
- Writes:
  - category_mapping.csv
  - canonicalized per-client transactions
  - features_per_client.csv  (monthly, ready for benefit scoring)

"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from rapidfuzz import process, fuzz


load_dotenv()


CANONICAL_CATEGORIES = [
    "Одежда и обувь",
    "Продукты питания",
    "Кафе и рестораны",
    "Медицина",
    "Авто",
    "Спорт",
    "Развлечения",
    "АЗС",
    "Кино",
    "Питомцы",
    "Книги",
    "Цветы",
    "Едим дома",
    "Смотрим дома",
    "Играем дома",
    "Косметика и Парфюмерия",
    "Подарки",
    "Ремонт дома",
    "Мебель",
    "Спа и массаж",
    "Ювелирные украшения",
    "Такси",
    "Отели",
    "Путешествия",
]


def llm_map_to_canon(
    raw_label: str, options: list[str], api_key: str, threshold: float = 0.60
):
    """
    Ask Gemini 2.5 Flash to map `raw_label` to exactly one of `options`.
    Returns dict with keys: canonical_category, confidence, method, reason
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    sys_msg = (
        "You convert noisy Russian purchase categories into ONE canonical category from the provided list. "
        "Return STRICT JSON with keys: canonical_category (string from the list or 'unknown'), "
        "confidence (0..1), reason (short). Do NOT add extra keys."
    )

    # We keep the prompt short and deterministic; model must choose from options only.
    prompt = {"raw_label": raw_label, "canonical_options": options}

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash", system_instruction=sys_msg
    )
    resp = model.generate_content(
        [prompt],
        generation_config=dict(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    text = resp.text or "{}"
    try:
        data = json.loads(text)
    except Exception:
        # Very defensive fallback
        return {
            "canonical_category": "unknown",
            "confidence": 0.0,
            "method": "llm",
            "reason": "json_parse_error",
        }

    canon = data.get("canonical_category", "unknown")
    conf = float(data.get("confidence", 0.0))
    if canon not in options:
        # If model proposed a non-list value, force unknown
        canon = "unknown"
    if conf < threshold:
        canon = "unknown"
    return {
        "canonical_category": canon,
        "confidence": conf,
        "method": "llm",
        "reason": data.get("reason", ""),
    }


def fuzzy_map_to_canon(raw_label: str, options: list[str], threshold: int = 87):
    """
    Use RapidFuzz to pick best canonical label by similarity.
    If score >= threshold, accept; else return unknown.
    """
    match = process.extractOne(
        raw_label,
        options,
        scorer=fuzz.WRatio,
    )
    if not match:
        return {"canonical_category": "unknown", "confidence": 0.0, "method": "fuzzy"}
    label, score, _ = match
    if score >= threshold:
        return {
            "canonical_category": label,
            "confidence": float(score) / 100.0,
            "method": "fuzzy",
        }
    return {
        "canonical_category": "unknown",
        "confidence": float(score) / 100.0,
        "method": "fuzzy",
    }


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_map_to_canon(raw_label: str, options: list[str]):
    raw_norm = normalize_text(raw_label)
    # direct hit (case sensitive first)
    if raw_norm in options:
        return {"canonical_category": raw_norm, "confidence": 1.0, "method": "exact"}
    # case-insensitive hit
    low = raw_norm.lower()
    low_map = {o.lower(): o for o in options}
    if low in low_map:
        return {
            "canonical_category": low_map[low],
            "confidence": 0.99,
            "method": "exact_ci",
        }
    return {"canonical_category": "unknown", "confidence": 0.0, "method": "exact"}


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: dict):
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ETL: discover categories across all clients
def list_client_files(data_dir: Path):
    tx_files = sorted(data_dir.glob("client_*_transactions_3m.csv"))
    tr_files = sorted(data_dir.glob("client_*_transfers_3m.csv"))
    # infer client codes from filenames
    tx_map, tr_map = {}, {}
    for p in tx_files:
        code = int(re.search(r"client_(\d+)_transactions_3m\.csv", p.name).group(1))
        tx_map[code] = p
    for p in tr_files:
        code = int(re.search(r"client_(\d+)_transfers_3m\.csv", p.name).group(1))
        tr_map[code] = p
    codes = sorted(set(tx_map) | set(tr_map))
    return codes, tx_map, tr_map


def discover_raw_categories(data_dir: Path, codes: list[int], tx_map: dict) -> set[str]:
    discovered = set()
    for code in codes:
        f = tx_map.get(code)
        if not f or not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "category" not in df.columns:
            continue
        df["category"] = df["category"].astype(str).map(normalize_text)
        discovered.update(df["category"].dropna().unique().tolist())
    return discovered


# Build master mapping
def build_mapping(
    raw_categories: set[str],
    canon: list[str],
    out_dir: Path,
    use_llm: bool = True,
    fuzzy_threshold: int = 87,
    llm_threshold: float = 0.60,
) -> pd.DataFrame:

    cache_path = out_dir / "taxonomy_cache.json"
    cache = load_cache(cache_path)

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if use_llm and not api_key:
        print("WARN: GOOGLE_API_KEY not set. LLM fallback will be skipped.")
        use_llm = False

    records = []
    for raw in sorted(raw_categories):
        raw_norm = normalize_text(raw)
        # cache hit?
        if raw_norm in cache:
            m = cache[raw_norm]
            records.append(
                {
                    "raw_category": raw_norm,
                    "canonical_category": m.get("canonical_category", "unknown"),
                    "method": m.get("method", "cache"),
                    "confidence": m.get("confidence", 0.0),
                    "reason": m.get("reason", ""),
                }
            )
            continue

        # exact
        m = exact_map_to_canon(raw_norm, canon)
        if m["canonical_category"] != "unknown":
            pass
        else:
            # fuzzy
            m = fuzzy_map_to_canon(raw_norm, canon, threshold=fuzzy_threshold)
            if m["canonical_category"] == "unknown" and use_llm:
                # LLM fallback
                try:
                    m = llm_map_to_canon(
                        raw_norm, canon, api_key, threshold=llm_threshold
                    )
                except Exception as e:
                    m = {
                        "canonical_category": "unknown",
                        "confidence": 0.0,
                        "method": "llm_error",
                        "reason": str(e),
                    }

        # persist
        cache[raw_norm] = m
        records.append(
            {
                "raw_category": raw_norm,
                "canonical_category": m["canonical_category"],
                "method": m["method"],
                "confidence": m["confidence"],
                "reason": m.get("reason", ""),
            }
        )

    # save cache + mapping CSV
    save_cache(cache_path, cache)
    mapping_df = pd.DataFrame(records)
    mapping_df.to_csv(out_dir / "category_mapping.csv", index=False)
    return mapping_df


# Apply mapping and build features
ONLINE_SET = {"Играем дома", "Смотрим дома", "Едим дома", "Кино"}
TRAVEL_SET = {"Такси", "Отели", "Путешествия"}
PREMIUM_BOOST_SET = {
    "Ювелирные украшения",
    "Косметика и Парфюмерия",
    "Кафе и рестораны",
}


def apply_mapping_to_tx(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mp = dict(zip(mapping["raw_category"], mapping["canonical_category"]))
    df = df.copy()
    df["category"] = df["category"].astype(str).map(normalize_text)
    df["canonical_category"] = df["category"].map(lambda x: mp.get(x, "unknown"))
    return df


def months_in_df(dt_series: pd.Series) -> int:
    return max(1, dt_series.dt.to_period("M").nunique())


def features_for_client(
    client_code: int, clients_df: pd.DataFrame, tx_df: pd.DataFrame, tr_df: pd.DataFrame
) -> dict:
    feat = {"client_code": int(client_code)}
    # profile
    prof = clients_df.loc[clients_df["client_code"] == client_code]
    if not prof.empty:
        row = prof.iloc[0].to_dict()
        feat.update(
            {
                "name": row.get("name", ""),
                "status": row.get("status", ""),
                "age": row.get("age", ""),
                "city": row.get("city", ""),
                "avg_monthly_balance_KZT": float(row.get("avg_monthly_balance_KZT", 0)),
            }
        )

    # transactions
    if not tx_df.empty and "date" in tx_df.columns:
        tx = tx_df.copy()
        tx["date"] = pd.to_datetime(tx["date"])
        tx["month"] = tx["date"].dt.to_period("M")
        m = months_in_df(tx["date"])
        feat["months_covered"] = m

        # totals
        feat["spend_total_3m"] = float(tx["amount"].sum())
        feat["spend_monthly_avg"] = float(tx["amount"].sum() / m)

        # by canonical category
        by_cat = (
            tx.groupby("canonical_category")["amount"]
            .sum()
            .sort_values(ascending=False)
        )
        top3 = list(by_cat.head(3).index)
        t1, t2, t3 = (top3 + ["", "", ""])[:3]
        feat["top1_cat"] = t1
        feat["top2_cat"] = t2
        feat["top3_cat"] = t3
        feat["top1_amt"] = float(by_cat.get(t1, 0.0))
        feat["top2_amt"] = float(by_cat.get(t2, 0.0))
        feat["top3_amt"] = float(by_cat.get(t3, 0.0))

        # themed spends
        feat["travel_spend_3m"] = float(
            tx[tx["canonical_category"].isin(TRAVEL_SET)]["amount"].sum()
        )
        feat["online_spend_3m"] = float(
            tx[tx["canonical_category"].isin(ONLINE_SET)]["amount"].sum()
        )
        feat["premium_boost_spend_3m"] = float(
            tx[tx["canonical_category"].isin(PREMIUM_BOOST_SET)]["amount"].sum()
        )
        # counts
        feat["taxi_trips_3m"] = int((tx["canonical_category"] == "Такси").sum())

        # currency coverage
        feat["non_kzt_spend_3m"] = (
            float(tx.loc[tx.get("currency", "KZT") != "KZT", "amount"].sum())
            if "currency" in tx.columns
            else 0.0
        )

    # transfers
    if not tr_df.empty and "date" in tr_df.columns:
        tr = tr_df.copy()
        tr["date"] = pd.to_datetime(tr["date"])
        m = months_in_df(tr["date"])
        # fx volumes
        feat["fx_volume_3m"] = float(
            tr[tr["type"].isin(["fx_buy", "fx_sell"])]["amount"].sum()
        )
        # invest volume
        feat["invest_volume_3m"] = float(
            tr[tr["type"].isin(["invest_out", "invest_in"])]["amount"].sum()
        )
        # atm
        atm = tr[(tr["type"] == "atm_withdrawal") & (tr["direction"] == "out")]
        feat["atm_withdrawals_count_3m"] = int(len(atm))
        feat["atm_withdrawals_amount_3m"] = float(atm["amount"].sum())
        # card_out
        co = tr[(tr["type"] == "card_out") & (tr["direction"] == "out")]
        feat["card_out_amount_3m"] = float(co["amount"].sum())
        feat["card_out_count_3m"] = int(len(co))
        # salary/income
        income = tr[tr["type"].isin(["salary_in", "stipend_in", "family_in"])]
        feat["income_amount_3m"] = float(income["amount"].sum())

    return feat


DATA_DIR = Path("./case_1")
OUT_DIR = Path("./data")
USE_LLM = True  # LLM fallback for taxonomy mapping
FUZZY_THRESHOLD = 87  # RapidFuzz accept threshold (0-100)
LLM_THRESHOLD = 0.60  # Min confidence to accept LLM mapping (0..1)


def main():
    data_dir = DATA_DIR
    out_dir = OUT_DIR
    use_llm = USE_LLM

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "canonicalized").mkdir(exist_ok=True)

    clients_path = data_dir / "clients.csv"
    if not clients_path.exists():
        raise FileNotFoundError(f"clients.csv not found at {clients_path}")
    clients = pd.read_csv(clients_path)

    # List client files
    codes, tx_map, tr_map = list_client_files(data_dir)
    print(f"Discovered {len(codes)} client codes")

    # Discover raw categories across all clients
    raw_cats = discover_raw_categories(data_dir, codes, tx_map)
    print(f"Discovered {len(raw_cats)} raw categories")

    # Build mapping (exact -> fuzzy -> LLM fallback)
    mapping_df = build_mapping(
        raw_categories=raw_cats,
        canon=CANONICAL_CATEGORIES,
        out_dir=out_dir,
        use_llm=use_llm,
        fuzzy_threshold=FUZZY_THRESHOLD,
        llm_threshold=LLM_THRESHOLD,
    )
    unknown_cnt = int((mapping_df["canonical_category"] == "unknown").sum())
    print(f"Mapping built. Unknowns: {unknown_cnt} / {len(mapping_df)}")

    # Canonicalize per-client + build features
    feature_rows = []
    for code in codes:
        # Load transactions
        txf = tx_map.get(code)
        if txf and txf.exists():
            tx = pd.read_csv(txf)
        else:
            tx = pd.DataFrame(
                columns=["date", "category", "amount", "currency", "client_code"]
            )

        # Load transfers
        trf = tr_map.get(code)
        if trf and trf.exists():
            tr = pd.read_csv(trf)
        else:
            tr = pd.DataFrame(
                columns=[
                    "date",
                    "type",
                    "direction",
                    "amount",
                    "currency",
                    "client_code",
                ]
            )

        # Types
        if "amount" in tx.columns:
            tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
        if "amount" in tr.columns:
            tr["amount"] = pd.to_numeric(tr["amount"], errors="coerce").fillna(0.0)

        # Apply taxonomy mapping
        if "category" in tx.columns:
            tx = apply_mapping_to_tx(tx, mapping_df)
        else:
            tx["canonical_category"] = "unknown"

        # Save canonicalized transactions
        tx_out = out_dir / "canonicalized" / f"client_{code}_transactions_3m_canon.csv"
        tx.to_csv(tx_out, index=False)

        # Build features
        feats = features_for_client(code, clients, tx, tr)
        feature_rows.append(feats)

    # Write features CSV
    feat_df = pd.DataFrame(feature_rows)
    feat_df.to_csv(out_dir / "features_per_client.csv", index=False)

    # Summary
    unknown_share = (mapping_df["canonical_category"] == "unknown").mean()
    print("Done.")
    print(f"- category_mapping.csv     -> {out_dir / 'category_mapping.csv'}")
    print(f"- features_per_client.csv  -> {out_dir / 'features_per_client.csv'}")
    print(f"- canonicalized tx dir     -> {out_dir / 'canonicalized'}")
    print(f"- Unknown category share   -> {unknown_share:.1%}")


if __name__ == "__main__":
    main()
