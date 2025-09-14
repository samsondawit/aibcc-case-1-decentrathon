from pathlib import Path
import pandas as pd


DATA_DIR = Path("./case_1")
OUT_DIR = Path("./data")
CANON_DIR = OUT_DIR / "canonicalized"
FEATURES_PATH = OUT_DIR / "features_per_client.csv"

# Product constants
TRAVEL_SET = {"Такси", "Отели", "Путешествия"}
ONLINE_SET = {"Играем дома", "Смотрим дома", "Едим дома", "Кино"}
PREMIUM_BOOST_SET = {
    "Ювелирные украшения",
    "Косметика и Парфюмерия",
    "Кафе и рестораны",
}


PREMIUM_BASE_LOW = 0.02  # < 1 000 000 tng
PREMIUM_BASE_MID = 0.03  # 1–6 млн tng
PREMIUM_BASE_HIGH = 0.04  # >= 6 млн tng
PREMIUM_CB_CAP = 100_000  # tng per month cap
ATM_SAVE_RATE = 0.01  # 1% proxy savings
ATM_SAVE_PER_TX = 200  # tng per withdrawal proxy
CARD_OUT_SAVE_RATE = 0.005  # 0.5% proxy savings
CREDIT_RATE = 0.10  # up to 10% on top-3 + online
TRAVEL_RATE = 0.04  # 4%
FX_SAVE_RATE = 0.007  # 0.7% better rate proxy
INV_SAVE_RATE = 0.003  # 0.3% commission proxy

APR_SAVING = 0.165  # 16.5%
APR_ACCUM = 0.155  # 15.5%
APR_MULTI = 0.145  # 14.5%

# Loan signal thresholds
LOW_BALANCE_CUTOFF = 300_000
BIG_PURCHASE_RATIO = 0.50  # any single tx > 50% of avg balance

# Fatigue / send-safety
BENEFIT_MIN_TO_SEND = 1_000  # tng per month

# Tie-breaker priority (stable order when benefits equal within epsilon)
TIE_BREAK_ORDER = [
    "Кредитная карта",
    "Премиальная карта",
    "Карта для путешествий",
    "Депозит Сберегательный (защита KDIF)",
    "Депозит Накопительный",
    "Депозит Мультивалютный (KZT/USD/RUB/EUR)",
    "Обмен валют",
    "Инвестиции",
    "Кредит наличными",
]

EPS = 1e-6

BASELINE_CB = 0.01  # assumed baseline debit cashback
CREDIT_CAP_MONTHLY = 30_000  # cap on monthly 10% benefit (tune)
CREDIT_MIN_ELIGIBLE_SPEND = 30_000
REALIZATION_MIN, REALIZATION_MAX = 0.30, 0.95
TRAVEL_REALIZATION_BASE = 0.40
PREMIUM_REALIZATION_LOW = 0.50
PREMIUM_REALIZATION_HIGH = 0.90


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def months_guard(x):
    try:
        m = int(x)
        return max(1, m)
    except Exception:
        return 3  # default


def base_rate_by_balance(avg_bal: float) -> float:
    if avg_bal >= 6_000_000:
        return PREMIUM_BASE_HIGH
    if avg_bal >= 1_000_000:
        return PREMIUM_BASE_MID
    return PREMIUM_BASE_LOW


def read_canon_tx(client_code: int) -> pd.DataFrame:
    f = CANON_DIR / f"client_{client_code}_transactions_3m_canon.csv"
    if f.exists():
        df = pd.read_csv(f)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    # fallback empty
    return pd.DataFrame(
        columns=["date", "canonical_category", "amount", "currency", "client_code"]
    )


def spend_sum(df: pd.DataFrame, cats: set[str]) -> float:
    if df.empty:
        return 0.0
    mask = df["canonical_category"].isin(cats)
    return float(df.loc[mask, "amount"].sum())


def max_tx_amount(df: pd.DataFrame) -> float:
    if df.empty or "amount" not in df:
        return 0.0
    return float(pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).max())


def pick_deposit(
    avg_bal: float,
    monthly_income: float,
    monthly_spend: float,
    fx_volume_3m: float,
    travel_spend_3m: float,
):
    """Return (deposit_name, monthly_benefit_kzt)."""
    # Simple behavioral chooser
    if monthly_income >= 0.5 * max(1.0, monthly_spend):
        apr = APR_ACCUM
        name = "Депозит Накопительный"
    elif (fx_volume_3m > 0) or (travel_spend_3m > 0):
        apr = APR_MULTI
        name = "Депозит Мультивалютный (KZT/USD/RUB/EUR)"
    else:
        apr = APR_SAVING
        name = "Депозит Сберегательный (защита KDIF)"
    monthly = apr * float(avg_bal) / 12.0
    return name, monthly


def credit_union_spend(df: pd.DataFrame, top3_cats: list[str]) -> float:
    """Union of top-3 categories and ONLINE_SET."""
    cats = set([c for c in top3_cats if c]) | ONLINE_SET
    return spend_sum(df, cats)


def credit_eligibility_score(
    monthly_income: float,
    monthly_spend: float,
    online_spend_pm: float,
    atm_cnt_pm: float,
    avg_bal: float,
) -> float:
    score = 0.0
    if monthly_income >= 120_000:
        score += 0.4  # steady inflow -> easy repayment/autopay
    if online_spend_pm >= 20_000:
        score += 0.2  # uses online services -> will capture 10%
    if monthly_spend >= 150_000:
        score += 0.2  # has enough volume to benefit
    if atm_cnt_pm > 2:
        score -= 0.2  # cash-heavy habit reduces card adoption
    if avg_bal < 50_000 and monthly_income == 0:
        score -= 0.3  # fragile cashflow -> weak adoption
    return clamp01(score)


def benefits_for_client_row(row: pd.Series, tx_df: pd.DataFrame) -> dict:
    m = months_guard(row.get("months_covered", 3))
    avg_bal = float(row.get("avg_monthly_balance_KZT", 0))
    spend_total_3m = float(row.get("spend_total_3m", 0))
    spend_monthly_avg = spend_total_3m / m if m else 0.0

    travel_spend_3m = float(row.get("travel_spend_3m", 0))
    premium_boost_spend_3m = float(row.get("premium_boost_spend_3m", 0))
    online_spend_3m = (
        float(row.get("online_spend_3m", 0)) if "online_spend_3m" in row else 0.0
    )

    fx_volume_3m = float(row.get("fx_volume_3m", 0))
    invest_volume_3m = float(row.get("invest_volume_3m", 0))
    non_kzt_spend_3m = float(row.get("non_kzt_spend_3m", 0))

    atm_cnt_3m = int(row.get("atm_withdrawals_count_3m", 0))
    atm_amt_3m = float(row.get("atm_withdrawals_amount_3m", 0))
    card_out_amt_3m = float(row.get("card_out_amount_3m", 0))

    income_amount_3m = float(row.get("income_amount_3m", 0))
    monthly_income = income_amount_3m / m if m else 0.0

    # Derived monthlys
    travel_pm = travel_spend_3m / m
    premium_boost_pm = premium_boost_spend_3m / m
    online_pm = online_spend_3m / m
    atm_cnt_pm = atm_cnt_3m / m if m else 0.0
    atm_amt_pm = atm_amt_3m / m
    card_out_amt_pm = card_out_amt_3m / m

    # Travel Card (incremental vs baseline)
    # eligible travel spend: taxi + hotels + travel
    travel_eligible_pm = travel_pm
    travel_realization = clamp01(
        TRAVEL_REALIZATION_BASE
        + 0.03 * min(12, row.get("taxi_trips_3m", 0) / max(1, m))
    )
    travel_inc_rate = max(0.0, TRAVEL_RATE - BASELINE_CB)  # extra vs baseline
    travel_benefit = travel_realization * travel_inc_rate * travel_eligible_pm

    # Premium Card (incremental vs baseline + ATM/transfer savings)
    base_rate = base_rate_by_balance(avg_bal)
    # incremental on boosted categories: (max(base, 0.04) - baseline)
    prem_inc_boost = max(base_rate, 0.04) - BASELINE_CB
    prem_inc_other = max(0.0, base_rate - BASELINE_CB)
    prem_cashback_inc = (
        prem_inc_other * max(0.0, spend_monthly_avg - premium_boost_pm)
        + prem_inc_boost * premium_boost_pm
    )
    prem_cashback_inc = min(prem_cashback_inc, PREMIUM_CB_CAP)

    atm_save = ATM_SAVE_RATE * atm_amt_pm + ATM_SAVE_PER_TX * atm_cnt_pm
    card_out_save = CARD_OUT_SAVE_RATE * card_out_amt_pm

    premium_realization = (
        PREMIUM_REALIZATION_HIGH if avg_bal >= 1_000_000 else PREMIUM_REALIZATION_LOW
    )
    premium_benefit = premium_realization * prem_cashback_inc + atm_save + card_out_save

    # Credit Card (incremental vs baseline + cap + realization)
    top3 = [row.get("top1_cat", ""), row.get("top2_cat", ""), row.get("top3_cat", "")]
    eligible_union_3m = credit_union_spend(tx_df, top3)  # 3m sum top-3 U ONLINE_SET
    eligible_union_pm = eligible_union_3m / m if m else 0.0
    eligible_union_pm = min(eligible_union_pm, spend_monthly_avg)  # safety

    # adoption likelihood
    cred_score = credit_eligibility_score(
        monthly_income, spend_monthly_avg, online_pm, atm_cnt_pm, avg_bal
    )
    credit_realization = (
        REALIZATION_MIN + (REALIZATION_MAX - REALIZATION_MIN) * cred_score
    )

    # if tiny eligible volume, down-weight realization
    if eligible_union_pm < CREDIT_MIN_ELIGIBLE_SPEND:
        credit_realization *= 0.6

    # incremental rate vs baseline (e.g., 10% - 1%)
    credit_inc_rate = max(0.0, CREDIT_RATE - BASELINE_CB)
    credit_gross = credit_inc_rate * eligible_union_pm
    credit_gross = min(credit_gross, CREDIT_CAP_MONTHLY)
    credit_benefit = credit_realization * credit_gross

    # FX / Multicurrency (keep simple, but monthlyize and use threshold)
    fx_volume_total_pm = (fx_volume_3m + non_kzt_spend_3m) / m if m else 0.0
    fx_realization = clamp01(
        0.4 + 0.4 * (fx_volume_total_pm >= 100_000)
    )  # more likely if sizable FX
    fx_benefit = fx_realization * FX_SAVE_RATE * fx_volume_total_pm

    # Deposits: pick best type then monthly reward
    deposit_name, deposit_benefit = pick_deposit(
        avg_bal, monthly_income, spend_monthly_avg, fx_volume_3m, travel_spend_3m
    )
    # small realization if balance is tiny (people don’t actually open)
    deposit_realization = (
        0.9 if avg_bal >= 1_000_000 else 0.6 if avg_bal >= 200_000 else 0.3
    )
    deposit_benefit *= deposit_realization

    # Investments
    invest_pm = invest_volume_3m / m if m else 0.0
    invest_realization = clamp01(0.3 + 0.5 * (avg_bal >= 200_000))
    invest_benefit = invest_realization * INV_SAVE_RATE * invest_pm

    # Cash Loan (rare; tiny placeholder)
    max_tx = max_tx_amount(tx_df)
    loan_signal = (avg_bal < LOW_BALANCE_CUTOFF) and (
        max_tx > BIG_PURCHASE_RATIO * max(1.0, avg_bal)
    )
    cash_loan_benefit = 1.0 if loan_signal else 0.0

    # Assemble
    benefits = {
        "Карта для путешествий": travel_benefit,
        "Премиальная карта": premium_benefit,
        "Кредитная карта": credit_benefit,
        "Обмен валют": fx_benefit,
        deposit_name: deposit_benefit,
        "Инвестиции": invest_benefit,
        "Кредит наличными": cash_loan_benefit,
    }
    return benefits


def stable_sort_products(benefits: dict) -> list[tuple[str, float]]:
    # Sort by benefit desc, then by TIE_BREAK_ORDER for stability
    order_index = {name: i for i, name in enumerate(TIE_BREAK_ORDER)}
    return sorted(
        benefits.items(), key=lambda kv: (-kv[1], order_index.get(kv[0], 999))
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Expected features at {FEATURES_PATH}. Run build_features.py first."
        )

    feats = pd.read_csv(FEATURES_PATH)
    rows = []
    top_rows = []

    for _, r in feats.iterrows():
        code = int(r["client_code"])
        tx_df = read_canon_tx(code)

        b = benefits_for_client_row(r, tx_df)
        ranked = stable_sort_products(b)

        # Wide row with all products (fill zero for non-picked deposit types)
        row = {"client_code": code}
        for name in TIE_BREAK_ORDER:
            row[name] = float(b.get(name, 0.0))
        # ensure whichever deposit was picked is included even if not in tie-order preset
        for k, v in b.items():
            if k not in row:
                row[k] = float(v)
        row["best_product"] = ranked[0][0]
        row["best_benefit_kzt"] = float(ranked[0][1])
        rows.append(row)

        # Top-4 skinny row
        top = {"client_code": code}
        for i, (name, val) in enumerate(ranked[:4], start=1):
            top[f"top{i}_product"] = name
            top[f"top{i}_benefit_kzt"] = float(val)
        top_rows.append(top)

    benefits_df = pd.DataFrame(rows)
    top4_df = pd.DataFrame(top_rows)

    benefits_path = OUT_DIR / "benefits_per_client.csv"
    top4_path = OUT_DIR / "top4_per_client.csv"

    benefits_df.to_csv(benefits_path, index=False)
    top4_df.to_csv(top4_path, index=False)

    print(f"- benefits_per_client.csv -> {benefits_path}")
    print(f"- top4_per_client.csv     -> {top4_path}")


if __name__ == "__main__":
    main()
