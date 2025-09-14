from __future__ import annotations
import pandas as pd
import base64, json
from pathlib import Path
from dotenv import load_dotenv
import os, re, json, time, random
from typing import Any, Dict, Tuple, Optional

load_dotenv()

DATA_DIR = Path("./case_1")
OUT_DIR = Path("./data")
CANON_DIR = OUT_DIR / "canonicalized"
TOP4_PATH = OUT_DIR / "top4_per_client.csv"
DEBUG_JSONL = OUT_DIR / "llm_copy_debug.jsonl"
FEATURES_PATH = OUT_DIR / "features_per_client.csv"
BENEFITS_PATH = OUT_DIR / "benefits_per_client.csv"

USE_LLM = True
LLM_MAX_RETRIES = 4
MIN_LEN, MAX_LEN = 180, 220
LLM_MODEL = "gemini-2.5-flash"
LLM_BACKOFF_SECS = [0.4, 0.8, 1.6, 2.5]

EMOJI_MAX = 1
ONE_EXCLAMATION_MAX = True
ALLOWED_CTA = [
    "Открыть",
    "Настроить",
    "Посмотреть",
    "Оформить сейчас",
    "Оформить карту",
    "Открыть вклад",
]

# Tone rules (enforced post-LLM too)
REQUIRE_VY_LOWER = True

# Category sets (used only for extra context if needed)
ONLINE_SET = {"Играем дома", "Смотрим дома", "Едим дома", "Кино"}
TRAVEL_SET = {"Такси", "Отели", "Путешествия"}
PREMIUM_BOOST_SET = {
    "Ювелирные украшения",
    "Косметика и Парфюмерия",
    "Кафе и рестораны",
}

DEFAULT_CTA_BY_PRODUCT = {
    "Кредитная карта": "Оформить карту",
    "Карта для путешествий": "Оформить сейчас",
    "Премиальная карта": "Оформить сейчас",
    "Депозит Сберегательный (защита KDIF)": "Открыть вклад",
    "Депозит Накопительный": "Открыть вклад",
    "Депозит Мультивалютный (KZT/USD/RUB/EUR)": "Открыть вклад",
    "Обмен валют": "Настроить",
    "Инвестиции": "Открыть",
    "Кредит наличными": "Посмотреть",
}


_FINISH_MAP = {
    0: "UNSPECIFIED",
    1: "STOP",
    2: "MAX_TOKENS",
    3: "SAFETY",
    4: "RECITATION",
    5: "OTHER",
    6: "BLOCKLIST",
    7: "PROHIBITED",
    8: "SPII",
}


def _fr_name(fr):
    if isinstance(fr, str):
        return fr.upper()
    try:
        return _FINISH_MAP.get(int(fr), str(fr))
    except:
        return str(fr)


def get_genai():
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    genai.configure(api_key=api_key)
    return genai


SYSTEM_INSTR = (
    "Вы — русскоязычный копирайтер. Пишите короткие персональные пуш-уведомления.\n"
    "СТРОГО соблюдайте правила:\n"
    "- На равных, доброжелательно; обращение на «вы» (строго в нижнем регистре) встречается хотя бы один раз.\n"
    "- 1 мысль, 1 призыв (CTA). Без воды и капса. 0–1 эмодзи.\n"
    "- Длина текста (без финального CTA): целитесь в 195±10 символов.\n"
    "- Деньги: «2 490 ₸» (пробелы между разрядами; валюта через пробел).\n"
    "- Дата при необходимости: дд.мм.гггг или «30 августа 2025».\n"
    "- Восклицательный знак: максимум один.\n"
    "- Никакого брендинга «наша/наш», без давления и дефицита.\n"
    'Выводите СТРОГО JSON по схеме: {"message":"...","cta":"..."} (cta — из списка). Никаких кодовых блоков/комментариев.\n'
)

TEMPLATES_HINTS = {
    "Карта для путешествий": "{name}, в {month_gen} у вас много поездок/такси. С тревел-картой часть расходов вернулась бы кешбэком. Хотите оформить?",
    "Премиальная карта": "{name}, у вас стабильно крупный остаток и траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас.",
    "Кредитная карта": "{name}, ваши топ-категории — {cat1}, {cat2}, {cat3}. Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. Оформить карту.",
    "Обмен валют": "{name}, вы часто платите в {fx_curr}. В приложении выгодный обмен и авто-покупка по целевому курсу. Настроить обмен.",
    "Депозит Сберегательный (защита KDIF)": "{name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. Открыть вклад.",
    "Депозит Накопительный": "{name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. Открыть вклад.",
    "Депозит Мультивалютный (KZT/USD/RUB/EUR)": "{name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. Открыть вклад.",
    "Инвестиции": "{name}, попробуйте инвестиции с низким порогом входа и без комиссий на старт. Открыть счёт.",
    "Кредит наличными": "{name}, если нужен запас на крупные траты — можно оформить кредит наличными с гибкими выплатами. Узнать доступный лимит.",
}


RUS_MONTHS_GEN = {
    1: "январе",
    2: "феврале",
    3: "марте",
    4: "апреле",
    5: "мае",
    6: "июне",
    7: "июле",
    8: "августе",
    9: "сентябре",
    10: "октябре",
    11: "ноябре",
    12: "декабре",
}


def month_genitive(dt) -> str:
    if pd.isna(dt):
        return ""
    return RUS_MONTHS_GEN.get(int(dt.month), "")


def fmt_kzt(x: float) -> str:
    try:
        n = int(round(float(x)))
    except Exception:
        n = 0
    s = f"{n:,}".replace(",", " ")
    return f"{s} ₸"


def approx(x: float, granularity: int = 100) -> int:
    if granularity <= 0:
        return int(round(x))
    return int(round(float(x) / granularity) * granularity)


def first_non_kzt_currency(tx_df: pd.DataFrame) -> Optional[str]:
    if "currency" not in tx_df.columns:
        return None
    cur = [c for c in tx_df["currency"].dropna().unique().tolist() if c != "KZT"]
    if not cur:
        return None
    return cur[0]


EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]"
)


def count_emoji(s: str) -> int:
    return len(EMOJI_RE.findall(s or ""))


def enforce_vy_form(s: str) -> str:
    s = re.sub(r"\bВы\b", "вы", s)  # downcase standalone 'Вы'
    low = " " + s.lower() + " "
    if not re.search(r"\bвы\b|\bвас\b|\bвам\b|\bвами\b|\bваш\w*\b", low):
        # place after name if present
        s = re.sub(r"^([^,]+,\s*)", r"\1вы ", s, count=1)
        if not re.search(r"\bвы\b|\bвас\b", s.lower()):
            s = "вы " + s[0].lower() + s[1:]
    return s


def ensure_cta(msg: str, cta: str, product: str) -> str:
    chosen = (
        cta
        if cta in ALLOWED_CTA
        else DEFAULT_CTA_BY_PRODUCT.get(product, ALLOWED_CTA[0])
    )
    # remove conjugated forms slipped inside
    msg = re.sub(r"\bОформите карту\b", "Оформить карту", msg)
    msg = re.sub(r"\bОткройте вклад\b", "Открыть вклад", msg)
    # append if missing
    if not any(msg.endswith(x) or msg.endswith(x + ".") for x in ALLOWED_CTA):
        if not msg.endswith((".", "…", "!", "?")):
            msg += "."
        msg = f"{msg.rstrip()} {chosen}."
    return msg


def strip_brand_words(s: str) -> str:
    # avoid "наша", "наш"
    s = (
        re.sub(r"\bнаша\b|\bнаш\b", "", s, flags=re.IGNORECASE)
        .replace("  ", " ")
        .strip()
    )
    return s


def sanitize_llm_text_to_json(text: str) -> Any:
    """
    Accepts messy LLM output and tries to get a JSON object with 'message'.
    - Strips code fences like ```json ... ```
    - Extracts first {...} or [...] block
    - json.loads
    - If list -> take first dict with 'message'
    """
    if text is None:
        raise ValueError("empty LLM response")

    t = text.strip()

    # strip code fences
    if t.startswith("```"):
        t = t.strip("`")
        # Remove possible 'json' tag lines
        t = re.sub(r"^json\s*", "", t, flags=re.IGNORECASE)

    # If text contains a JSON object/array within other text, extract it
    # Try object
    mobj = re.search(r"\{[\s\S]*\}", t)
    if mobj:
        candidate = mobj.group(0)
    else:
        # Try array
        marr = re.search(r"\[[\s\S]*\]", t)
        candidate = marr.group(0) if marr else t

    data = json.loads(candidate)

    # If list -> pick first dict with 'message'
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "message" in item:
                return item
        # wrap as dict
        return {"message": str(data[0]) if data else ""}

    if isinstance(data, dict):
        return data

    # fallback
    return {"message": str(data)}


def validate_message(msg: str) -> Tuple[bool, str]:
    if not isinstance(msg, str) or not msg.strip():
        return False, "empty"

    s = msg.strip()

    # Length
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False, f"length {len(s)} not in [{MIN_LEN},{MAX_LEN}]"

    # One CTA at end (best-effort): endswith allowed CTA or contains one and ends with period
    has_cta = any(s.endswith(cta + ".") or s.endswith(cta) for cta in ALLOWED_CTA)
    if not has_cta:
        # try to detect CTA elsewhere
        has_any = any(cta in s for cta in ALLOWED_CTA)
        if not has_any:
            return False, "cta_missing"

    # Exclamation marks
    if ONE_EXCLAMATION_MAX and s.count("!") > 1:
        return False, "too_many_exclamations"

    # Emoji count
    if count_emoji(s) > EMOJI_MAX:
        return False, "too_many_emoji"

    # Avoid CAPS shouting (heuristic: long all-caps word)
    if re.search(r"\b[А-ЯЁ]{4,}\b", s):
        return False, "caps_shouting"

    # Must contain 'вы' (lowercase) at least once (soft check)
    if REQUIRE_VY_LOWER and (" вы " not in (" " + s.lower() + " ")):
        # tolerate if 'вас' present (still 'вы' form)
        if " вас " not in (" " + s.lower() + " "):
            return False, "no_vy_form"

    return True, "ok"


def lint_and_format(s: str) -> str:
    if not s:
        return s

    # normalize currency: ensure space before ₸ and spacing inside number
    s = re.sub(r"(\d)\s*₸", r"\1 ₸", s)

    # Replace non-breaking spaces or multiple spaces with single regular spaces
    s = re.sub(r"[ \u00A0]+", " ", s).strip()

    # Keep max one exclamation
    if ONE_EXCLAMATION_MAX:
        parts = s.split("!")
        if len(parts) > 2:
            s = "!".join(parts[:2])  # keep first !

    # Trim if slightly over max, prefer removing trailing period
    if len(s) > MAX_LEN:
        s = s[:MAX_LEN].rstrip(" ,;")

    return s


def write_debug(payload: Dict[str, Any]):
    with open(DEBUG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# CONTEXT BUILDER
def read_canon_tx(client_code: int) -> pd.DataFrame:
    f = CANON_DIR / f"client_{client_code}_transactions_3m_canon.csv"
    if f.exists():
        df = pd.read_csv(f)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return pd.DataFrame(
        columns=["date", "canonical_category", "amount", "currency", "client_code"]
    )


def build_context(
    product: str, feat_row: pd.Series, tx_df: pd.DataFrame, benefits_row: pd.Series
) -> Dict[str, Any]:
    # basics
    name = str(feat_row.get("name", "Клиент")).strip() or "Клиент"
    status = str(feat_row.get("status", "")).strip()
    city = str(feat_row.get("city", "")).strip()
    avg_bal = float(feat_row.get("avg_monthly_balance_KZT", 0))
    months = int(max(1, feat_row.get("months_covered", 3)))

    # dates
    last_date = pd.to_datetime(tx_df["date"].max()) if "date" in tx_df else pd.NaT
    month_gen = month_genitive(last_date) if pd.notna(last_date) else ""

    # categories
    top1 = str(feat_row.get("top1_cat", "")).strip()
    top2 = str(feat_row.get("top2_cat", "")).strip()
    top3 = str(feat_row.get("top3_cat", "")).strip()
    cat1, cat2, cat3 = (top1 or "…", top2 or "…", top3 or "…")

    # product benefit (monthly)
    benefit = 0.0
    if product in benefits_row:
        benefit = float(benefits_row[product] or 0.0)
    benefit_rounded = fmt_kzt(approx(benefit, 100))

    # extra signals
    taxi_cnt = (
        int((tx_df["canonical_category"] == "Такси").sum())
        if "canonical_category" in tx_df
        else 0
    )
    taxi_amt = (
        float(tx_df.loc[tx_df["canonical_category"] == "Такси", "amount"].sum())
        if "canonical_category" in tx_df
        else 0.0
    )
    taxi_amt_s = fmt_kzt(taxi_amt)
    fx_curr = first_non_kzt_currency(tx_df) or "валюте"

    return {
        "name": name,
        "status": status,
        "city": city,
        "avg_balance_kzt": avg_bal,
        "month_gen": month_gen,
        "cat1": cat1,
        "cat2": cat2,
        "cat3": cat3,
        "benefit_per_month_kzt": benefit_rounded,
        "taxi_cnt": taxi_cnt,
        "taxi_amt": taxi_amt_s,
        "fx_curr": fx_curr,
        "product": product,
        "template_hint": TEMPLATES_HINTS.get(product, ""),
    }


# LLM CALL WITH RETRIES + VALIDATION
def _compose_prompt_text(context: dict, tight: bool = False) -> str:
    allowed = json.dumps(ALLOWED_CTA, ensure_ascii=False)
    target_len = "195±10" if not tight else "190±10"
    ctx = _slim_context(context, tight=tight)
    return (
        SYSTEM_INSTR
        + "\n\nКонтекст JSON (используйте только эти факты и суммы; ничего не придумывайте):\n"
        + json.dumps(ctx, ensure_ascii=False)
        + "\n\nallowed_cta: "
        + allowed
        + "\n\nТребования к ответу:"
        f"\n- Длина сообщения (без CTA) ≈ {target_len} символов."
        "\n- НЕ вставляйте CTA внутрь текста. CTA выберите ОДИН из allowed_cta (точная форма, без спряжения)."
        '\n- Верните СТРОГО JSON: {"message":"...", "cta":"<одно из allowed_cta>"} — без кода/комментариев.'
        "\n\nИзбегайте ошибок:"
        "\n- Не пишите «наша карта»; не меняйте форму CTA; не ставьте 2+ эмодзи; не превышайте длину."
    )


def extract_raw_json_from_resp(resp) -> tuple[str, str]:
    # 0) quick accessor
    try:
        if getattr(resp, "text", None):
            return resp.text, "STOP"
    except Exception:
        pass

    try:
        rd = resp.to_dict()
    except Exception:
        rd = {}

    cands = rd.get("candidates") or []
    if not cands:
        pf = rd.get("promptFeedback") or {}
        br = pf.get("blockReason") or "UNKNOWN"
        return "", f"BLOCKED:{br}"

    c = cands[0]
    fr = _fr_name(c.get("finishReason") or c.get("finish_reason") or 0)
    parts = ((c.get("content") or {}).get("parts")) or []

    for p in parts:
        if p.get("text"):
            return p["text"], fr
    for p in parts:
        if "inline_data" in p:
            mime = (p["inline_data"].get("mime_type") or "").lower()
            if "json" in mime and "data" in p["inline_data"]:
                raw = base64.b64decode(p["inline_data"]["data"]).decode(
                    "utf-8", "ignore"
                )
                return raw, fr

    return "", fr


def _slim_context(ctx: dict, tight: bool) -> dict:
    keep = [
        "name",
        "status",
        "city",
        "month_gen",
        "cat1",
        "cat2",
        "cat3",
        "benefit_per_month_kzt",
        "product",
        "taxi_cnt",
        "taxi_amt",
        "fx_curr",
    ]
    slim = {k: ctx.get(k) for k in keep}
    if tight:
        for k in ["taxi_cnt", "taxi_amt", "city"]:
            slim.pop(k, None)
    return slim


def _response_schema():
    # JSON schema with enum for CTA — model cannot invent anything else
    return {
        "type": "OBJECT",
        "properties": {
            "message": {"type": "STRING"},
            "cta": {"type": "STRING", "enum": ALLOWED_CTA},
        },
        "required": ["message", "cta"],
    }


def stretch_if_short(s: str, ctx: dict) -> str:
    # If too short, add one concise clause BEFORE the CTA (keep single CTA)
    if len(s) >= MIN_LEN:
        return s
    # find CTA position
    cta_pos = min((s.find(cta) for cta in ALLOWED_CTA if cta in s), default=-1)
    benefit = ctx.get("benefit_per_month_kzt", "")
    add = f" Вернётся ≈ {benefit} ежемесячно."
    if cta_pos > 0:
        s = s[:cta_pos].rstrip(". ") + add + " " + s[cta_pos:]
    else:
        s = s + add
    return s


def assemble_final_message(data: dict, ctx: dict) -> str:
    msg = (data.get("message") or "").strip()
    cta = (data.get("cta") or "").strip()
    product = ctx.get("product", "")

    # basic assembly + lint
    msg = lint_and_format(msg)
    msg = strip_brand_words(msg)
    msg = enforce_vy_form(msg)
    msg = ensure_cta(msg, cta, product)

    # punctuation, emoji, and length tuning
    if ONE_EXCLAMATION_MAX and msg.count("!") > 1:
        parts = msg.split("!")
        msg = "!".join(parts[:2])  # keep first !
    if count_emoji(msg) > EMOJI_MAX:
        # remove extras
        extra = count_emoji(msg) - EMOJI_MAX
        msg = re.sub(EMOJI_RE, "", msg, count=extra)

    msg = stretch_if_short(msg, ctx)
    return msg


def generate_with_llm(context: Dict[str, Any]) -> str:
    if not USE_LLM:
        raise RuntimeError("LLM disabled by config")
    genai = get_genai()
    model = genai.GenerativeModel(model_name=LLM_MODEL, system_instruction=SYSTEM_INSTR)

    last_issues = []  # accumulate validator issues to show the model

    for attempt in range(LLM_MAX_RETRIES):
        if attempt > 0:
            delay = LLM_BACKOFF_SECS[min(attempt - 1, len(LLM_BACKOFF_SECS) - 1)]
            delay *= 0.9 + 0.2 * random.random()
            time.sleep(delay)

        tight = attempt >= 1
        prompt_text = _compose_prompt_text(context, tight=tight)
        if last_issues:
            prompt_text += (
                "\n\nИсправьте ВСЕ ошибки: "
                + "; ".join(last_issues)
                + '\nВерните СТРОГО JSON {"message":"...","cta":"..."}.'
            )

        # parts payload; fallback to plain string
        try:
            payload = [{"role": "user", "parts": [{"text": prompt_text}]}]
            resp = model.generate_content(
                payload,
                generation_config=dict(
                    response_mime_type="application/json",
                    response_schema=_response_schema(),
                    temperature=0.2,
                ),
            )
        except Exception:
            resp = model.generate_content(
                prompt_text,
                generation_config=dict(
                    response_mime_type="application/json",
                    response_schema=_response_schema(),
                    temperature=0.2,
                ),
            )

        raw, fr = extract_raw_json_from_resp(resp)
        if not raw:
            write_debug(
                {"stage": "llm_no_parts", "attempt": attempt + 1, "finish_reason": fr}
            )
            last_issues = [
                f"нет контента (finish_reason={fr})",
                "строгий JSON",
                "cta из allowed_cta",
            ]
            continue

        try:
            data = sanitize_llm_text_to_json(raw)
        except Exception as e:
            msg = f"JSON разметка повреждена ({str(e)[:120]}). Уберите кодовые блоки/текст, верните чистый JSON."
            last_issues = [msg]
            write_debug(
                {
                    "stage": "sanitize_error",
                    "attempt": attempt + 1,
                    "error": msg,
                    "raw": raw,
                }
            )
            continue

        if not isinstance(data, dict) or "message" not in data or "cta" not in data:
            last_issues = ["нет полей message|cta по схеме"]
            write_debug(
                {"stage": "schema_failed", "attempt": attempt + 1, "data": data}
            )
            continue
        if data["cta"] not in ALLOWED_CTA:
            last_issues = [f"cta не из списка allowed_cta: {data['cta']}"]
            write_debug({"stage": "cta_failed", "attempt": attempt + 1, "data": data})
            continue

        # assemble, auto-repair, validate
        msg = assemble_final_message(data, context)
        ok, reason = validate_message(msg)
        if ok:
            write_debug(
                {
                    "stage": "ok",
                    "attempt": attempt + 1,
                    "finish_reason": fr,
                    "output": {"message": msg},
                }
            )
            return msg

        # tell the model EXACTLY what failed
        last_issues = []
        if reason.startswith("length"):
            # extract numbers to give precise hint
            last_issues.append(f"длина {len(msg)} вне 180–220; нужно 190–210 до CTA")
        elif reason == "cta_missing":
            last_issues.append("CTA отсутствует; выберите из allowed_cta")
        elif reason == "no_vy_form":
            last_issues.append("добавьте «вы» или «вас» в нижнем регистре")
        elif reason == "too_many_exclamations":
            last_issues.append("не больше одного восклицательного знака")
        elif reason == "too_many_emoji":
            last_issues.append("не больше одного эмодзи")
        elif reason == "caps_shouting":
            last_issues.append("не используйте КАПС")

        write_debug(
            {
                "stage": "validation_failed",
                "attempt": attempt + 1,
                "reason": reason,
                "candidate": msg,
            }
        )

    raise RuntimeError(
        f"LLM failed after {LLM_MAX_RETRIES} attempts; last_error={'; '.join(last_issues) or 'unknown'}"
    )


# DETERMINISTIC FALLBACK
def fallback_copy(context: Dict[str, Any]) -> str:
    name = context["name"]
    month_gen = context["month_gen"] or ""
    cat1, cat2, cat3 = context["cat1"], context["cat2"], context["cat3"]
    benefit = context["benefit_per_month_kzt"]
    product = context["product"]
    taxi_cnt = context["taxi_cnt"]
    taxi_amt = context["taxi_amt"]
    fx_curr = context["fx_curr"]

    if product == "Кредитная карта":
        s = f"{name}, ваши топ-категории — {cat1}, {cat2}, {cat3}. Кредитная карта даст до 10% в любимых категориях и на онлайн-сервисы. Оформить карту."
    elif product == "Карта для путешествий":
        s = f"{name}, в {month_gen} у вас {taxi_cnt} поездок на такси на {taxi_amt}. С тревел-картой вернулась бы часть расходов — примерно {benefit}. Открыть."
    elif product == "Премиальная карта":
        s = f"{name}, у вас стабильный остаток и частые траты. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас."
    elif product.startswith("Депозит"):
        s = f"{name}, свободные средства можно разместить под доход {benefit} в месяц. Удобно копить и не отвлекаться. Открыть вклад."
    elif product == "Обмен валют":
        s = f"{name}, часто платите в {fx_curr}. В приложении выгодный обмен и авто-покупка по целевому курсу. Настроить."
    elif product == "Инвестиции":
        s = f"{name}, попробуйте инвестиции: низкий порог и без комиссий на старт. Открыть."
    else:
        s = f"{name}, если нужен запас на крупные траты — можно оформить кредит наличными с гибкими выплатами. Узнать доступный лимит."

    s = lint_and_format(s)
    # If still out of bounds, do a minimal cut
    if len(s) > MAX_LEN:
        s = s[:MAX_LEN].rstrip(" ,;.")
    return s


# MAIN
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_JSONL.write_text("")

    # Load inputs
    if (
        not FEATURES_PATH.exists()
        or not TOP4_PATH.exists()
        or not BENEFITS_PATH.exists()
    ):
        raise FileNotFoundError(
            "Missing build_features and build_benefits outputs. Ensure features_per_client.csv, top4_per_client.csv, benefits_per_client.csv exist."
        )

    feats = pd.read_csv(FEATURES_PATH)
    top4 = pd.read_csv(TOP4_PATH)
    benefits = pd.read_csv(BENEFITS_PATH)

    # Build index for quick lookup
    top1_map = dict(zip(top4["client_code"], top4["top1_product"]))
    # We'll also use full benefits row to fetch product benefit by name
    benefits_index = benefits.set_index("client_code")

    outputs = []
    for _, fr in feats.iterrows():
        code = int(fr["client_code"])
        product = top1_map.get(code, "")
        if not product:
            continue

        tx_df = read_canon_tx(code)
        context = build_context(product, fr, tx_df, benefits_index.loc[code])

        # Try LLM with retries; if fails, fallback
        try:
            msg = generate_with_llm(context) if USE_LLM else fallback_copy(context)
        except Exception as e:
            write_debug(
                {"stage": "fallback_triggered", "client_code": code, "error": str(e)}
            )
            msg = fallback_copy(context)

        outputs.append(
            {"client_code": code, "product": product, "push_notification": msg}
        )

    out_csv = OUT_DIR / "personalized_pushes.csv"
    pd.DataFrame(outputs).to_csv(out_csv, index=False)

    print(f"- personalized_pushes.csv -> {out_csv}")
    print(f"- debug log -> {DEBUG_JSONL}")


if __name__ == "__main__":
    main()
