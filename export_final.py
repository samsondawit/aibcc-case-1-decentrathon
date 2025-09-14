from __future__ import annotations
import pandas as pd
from pathlib import Path

OUT_DIR = Path("./data")
SUBMIT_DIR = OUT_DIR / "submissions"
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = OUT_DIR / "features_per_client.csv"
TOP4_PATH = OUT_DIR / "top4_per_client.csv"
BEN_PATH = OUT_DIR / "benefits_per_client.csv"
BEN_DEBUG_PATH = OUT_DIR / "benefits_debug_per_client.csv"
PUSHES_PATH = OUT_DIR / "personalized_pushes.csv"

# policy knobs
BENEFIT_MIN_TO_SEND = 1_000  # suppress if best benefit below this
MIN_LEN, MAX_LEN = 180, 220  # validation window for push text
ALLOWED_CTA = {
    "Открыть",
    "Настроить",
    "Посмотреть",
    "Оформить сейчас",
    "Оформить карту",
    "Открыть вклад",
}


def _validate_push(msg: str) -> tuple[bool, str]:
    if not isinstance(msg, str) or not msg.strip():
        return False, "empty"
    s = msg.strip()
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False, f"length {len(s)} not in [{MIN_LEN},{MAX_LEN}]"
    if not any(s.endswith(cta) or s.endswith(cta + ".") for cta in ALLOWED_CTA):
        return False, "cta_missing_or_not_at_end"
    return True, "ok"


def main():
    feats = pd.read_csv(FEATURES_PATH)
    top4 = pd.read_csv(TOP4_PATH)
    ben = pd.read_csv(BEN_PATH)
    ben_dbg = pd.read_csv(BEN_DEBUG_PATH) if BEN_DEBUG_PATH.exists() else pd.DataFrame()
    pushes = pd.read_csv(PUSHES_PATH)

    # join
    df = (
        feats[["client_code", "name", "status", "avg_monthly_balance_KZT"]]
        .merge(top4, on="client_code", how="left")
        .merge(
            ben[["client_code", "best_product", "best_benefit_kzt"]],
            on="client_code",
            how="left",
        )
        .merge(pushes, on="client_code", how="left", suffixes=("", ""))
    )

    # sanity: align product from pushes to our top1 if mismatched/empty
    df["product_final"] = df["top1_product"].where(
        df["product"].isna() | (df["product"] != df["top1_product"]), df["product"]
    )
    df["product_final"] = df["product_final"].fillna(df["top1_product"])

    # send/no-send policy
    df["send"] = (df["best_benefit_kzt"].fillna(0) >= BENEFIT_MIN_TO_SEND) & df[
        "push_notification"
    ].notna()

    # minimal validation of push
    df["push_ok"], df["push_issue"] = zip(
        *df["push_notification"].fillna("").map(_validate_push)
    )
    df.loc[~df["push_ok"], "send"] = False
    df.loc[~df["push_ok"], "push_issue"] = df["push_issue"].replace(
        "", "invalid_unknown"
    )

    # final submission (only rows to send)
    sub = (
        df.loc[df["send"], ["client_code", "product_final", "push_notification"]]
        .rename(columns={"product_final": "product"})
        .sort_values("client_code")
    )

    # richer debug (why/what)
    dbg_cols = [
        "client_code",
        "name",
        "status",
        "avg_monthly_balance_KZT",
        "top1_product",
        "top1_benefit_kzt",
        "top2_product",
        "top2_benefit_kzt",
        "top3_product",
        "top3_benefit_kzt",
        "top4_product",
        "top4_benefit_kzt",
        "best_product",
        "best_benefit_kzt",
        "product_final",
        "send",
        "push_ok",
        "push_issue",
        "push_notification",
    ]
    debug = df[dbg_cols]
    if not ben_dbg.empty:
        debug = debug.merge(ben_dbg, on="client_code", how="left")

    # write
    final_csv = SUBMIT_DIR / "pushes_final.csv"  # exact 3 columns for submission
    debug_csv = SUBMIT_DIR / "pushes_with_debug.csv"  # everything useful to inspect
    sub.to_csv(final_csv, index=False)
    debug.to_csv(debug_csv, index=False)

    # quick print
    print(f"- final CSV (3 cols)  -> {final_csv}")
    print(f"- debug CSV (rich)    -> {debug_csv}")
    print(f"- rows sent / total   -> {len(sub)} / {len(df)}")


if __name__ == "__main__":
    main()
