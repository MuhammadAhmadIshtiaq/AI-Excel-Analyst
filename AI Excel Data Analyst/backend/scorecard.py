"""
scorecard.py — Universal Intelligent Scorecard
Detects dataset domain first, then scores accordingly.
Completely isolated from engine.py — only reuses call_llm().
"""

import difflib
import pandas as pd
from engine import call_llm


# ─────────────────────────────────────────────
# STEP 1: DOMAIN DETECTOR
# Identifies what kind of dataset this is
# ─────────────────────────────────────────────

DOMAIN_KEYWORDS = {
    "sales": [
        "sales", "revenue", "profit", "cost", "price", "quantity", "discount",
        "amount", "unit", "product", "customer", "region", "channel", "payment",
        "order", "invoice", "margin", "turnover", "sold", "purchase"
    ],
    "health": [
        "age", "gender", "bmi", "weight", "height", "blood", "pressure", "heart",
        "sleep", "stress", "anxiety", "depression", "mental", "physical", "activity",
        "calories", "exercise", "medical", "diagnosis", "symptom", "health"
    ],
    "social_media": [
        "social", "media", "instagram", "tiktok", "facebook", "twitter", "platform",
        "screen", "addiction", "usage", "hours", "followers", "likes", "engagement",
        "posts", "interaction", "online"
    ],
    "hr": [
        "employee", "salary", "department", "hire", "attrition", "performance",
        "tenure", "promotion", "manager", "role", "job", "position", "workforce",
        "leave", "absentee", "headcount"
    ],
    "education": [
        "student", "grade", "score", "marks", "exam", "course", "subject",
        "attendance", "academic", "gpa", "pass", "fail", "university", "school",
        "teacher", "performance", "result"
    ],
    "ecommerce": [
        "cart", "checkout", "shipping", "delivery", "return", "refund", "sku",
        "listing", "seller", "buyer", "rating", "review", "category", "brand",
        "warehouse", "inventory", "stock"
    ],
    "finance": [
        "stock", "portfolio", "asset", "liability", "equity", "dividend", "interest",
        "loan", "credit", "debit", "balance", "transaction", "bank", "investment",
        "market", "trade", "fund", "expense", "budget"
    ],
}

def detect_domain(df: pd.DataFrame) -> str:
    """
    Scores each domain by keyword overlap with column names + sample values.
    Returns the best matching domain or 'general'.
    """
    # Build a text blob from column names + sample string values
    col_text = " ".join(df.columns).lower()
    sample_text = " ".join(
        df.select_dtypes(include="object").head(10).astype(str).values.flatten()
    ).lower()
    full_text = col_text + " " + sample_text

    domain_scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in full_text)
        domain_scores[domain] = score

    best_domain = max(domain_scores, key=domain_scores.get)
    best_score  = domain_scores[best_domain]

    # social_media is a sub-domain of health — merge if both score high
    if best_domain == "health" and domain_scores.get("social_media", 0) >= 2:
        best_domain = "social_media"

    print(f"[Scorecard] Domain scores: {domain_scores} → selected: {best_domain} (score={best_score})")
    return best_domain if best_score >= 2 else "general"


# ─────────────────────────────────────────────
# STEP 2: COLUMN RESOLVER  (fuzzy match)
# Maps actual column names to concept labels
# ─────────────────────────────────────────────

CONCEPT_KEYWORDS = {
    # Sales / Finance
    "revenue":   ["sales", "revenue", "amount", "sales_amount", "turnover", "income", "receipts"],
    "cost":      ["cost", "costs", "expense", "unit_cost", "cogs", "expenditure", "overhead"],
    "profit":    ["profit", "net", "margin", "earnings", "net_profit", "net_income"],
    "quantity":  ["quantity", "qty", "units", "sold", "volume", "count", "quantity_sold"],
    "discount":  ["discount", "rebate", "reduction", "promo"],
    "price":     ["price", "unit_price", "rate", "mrp", "selling_price"],
    "date":      ["date", "time", "month", "year", "period", "sale_date", "created"],
    "region":    ["region", "area", "zone", "territory", "location", "city", "country"],
    "category":  ["category", "product_category", "type", "segment", "group", "class"],
    # Health / Social Media
    "age":              ["age", "years"],
    "gender":           ["gender", "sex"],
    "sleep":            ["sleep", "sleep_hours", "rest"],
    "stress":           ["stress", "stress_level"],
    "anxiety":          ["anxiety", "anxiety_level"],
    "depression":       ["depression", "depression_label", "mood"],
    "addiction":        ["addiction", "addiction_level", "dependency"],
    "screen_time":      ["screen", "screen_time", "screen_time_before_sleep"],
    "social_hours":     ["social_media_hours", "daily_social_media_hours", "usage_hours"],
    "physical":         ["physical", "activity", "physical_activity", "exercise"],
    "academic":         ["academic", "academic_performance", "gpa", "grade", "score"],
    "platform":         ["platform", "platform_usage", "app", "social_platform"],
    # HR
    "salary":           ["salary", "wage", "compensation", "pay", "ctc"],
    "attrition":        ["attrition", "turnover", "churn", "left", "resigned"],
    "department":       ["department", "dept", "division", "team"],
    "performance":      ["performance", "rating", "appraisal", "kpi"],
    # Education
    "marks":            ["marks", "score", "grade", "result", "gpa", "percentage"],
    "attendance":       ["attendance", "present", "absent"],
    "subject":          ["subject", "course", "module"],
}

def fuzzy_resolve_columns(df: pd.DataFrame) -> dict:
    """Maps actual column names to concept labels using fuzzy matching."""
    matched  = {}
    used     = set()
    cols_map = {col.lower().replace(" ", "_"): col for col in df.columns}

    for concept, keywords in CONCEPT_KEYWORDS.items():
        best_col   = None
        best_score = 0.0

        for col_lower, col_original in cols_map.items():
            if col_original in used:
                continue
            for kw in keywords:
                score = difflib.SequenceMatcher(None, col_lower, kw).ratio()
                if score > best_score and score > 0.55:
                    best_score = score
                    best_col   = col_original

        if best_col:
            matched[concept] = best_col
            used.add(best_col)

    return matched


# ─────────────────────────────────────────────
# STEP 3: DOMAIN SCORECARDS
# Each returns {score, grade, metrics, passed, flags, advice}
# ─────────────────────────────────────────────

def _grade(score: int) -> str:
    return "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D"


# ── SALES / ECOMMERCE ─────────────────────────
def score_sales(df: pd.DataFrame, col: dict) -> dict:
    score, passed, flags, metrics = 0, [], [], {}

    rev_col  = col.get("revenue")
    cost_col = col.get("cost")
    qty_col  = col.get("quantity")
    disc_col = col.get("discount")
    price_col = col.get("price")
    date_col = col.get("date")

    # Revenue analysis
    if rev_col and rev_col in df.columns:
        total = df[rev_col].sum()
        metrics["total_revenue"] = round(float(total), 2)

        vol = df[rev_col].std() / df[rev_col].mean() if df[rev_col].mean() else 0
        metrics["revenue_volatility"] = round(float(vol), 4)

        if vol < 0.20:
            score += 20; passed.append("✅ Stable revenue stream")
        else:
            flags.append("📉 High revenue volatility — inconsistent sales")

        # Growth (needs date)
        if date_col and date_col in df.columns:
            try:
                df_s  = df.sort_values(date_col)
                h     = len(df_s) // 2
                g     = (df_s.iloc[h:][rev_col].sum() - df_s.iloc[:h][rev_col].sum()) / (df_s.iloc[:h][rev_col].sum() or 1)
                metrics["revenue_growth_pct"] = round(float(g * 100), 2)
                if g > 0.10:   score += 25; passed.append("✅ Strong revenue growth (>10%)")
                elif g > 0:    score += 10; passed.append("🟡 Moderate revenue growth")
                else:          flags.append("🚨 Revenue declining")
            except Exception:
                pass

    # Margin
    if rev_col and cost_col and rev_col in df.columns and cost_col in df.columns:
        rev  = df[rev_col].sum()
        cost = df[cost_col].sum()
        margin = (rev - cost) / rev if rev else 0
        metrics["profit_margin_pct"] = round(float(margin * 100), 2)
        metrics["total_cost"]        = round(float(cost), 2)
        if margin > 0.25:   score += 25; passed.append("✅ Healthy profit margin (>25%)")
        elif margin > 0.10: score += 12; passed.append("🟡 Acceptable margin (10–25%)")
        else:               flags.append("⚠️ Thin profit margin — review cost structure")

    # Discount impact
    if disc_col and disc_col in df.columns and rev_col and rev_col in df.columns:
        avg_disc = df[disc_col].mean()
        metrics["avg_discount_pct"] = round(float(avg_disc * 100 if avg_disc < 1 else avg_disc), 2)
        if avg_disc < 0.10 or avg_disc < 10:
            score += 15; passed.append("✅ Discounts well controlled")
        else:
            flags.append("⚠️ High average discount — may be eroding margin")

    # Quantity
    if qty_col and qty_col in df.columns:
        metrics["total_units_sold"] = int(df[qty_col].sum())
        score += 10; passed.append("✅ Quantity data available")

    # Price vs cost
    if price_col and cost_col and price_col in df.columns and cost_col in df.columns:
        markup = (df[price_col].mean() - df[cost_col].mean()) / df[cost_col].mean() if df[cost_col].mean() else 0
        metrics["avg_markup_pct"] = round(float(markup * 100), 2)
        if markup > 0.30: score += 5; passed.append("✅ Healthy markup (>30%)")

    score = min(score, 100)
    advice = []
    for f in flags:
        if "declining"  in f.lower(): advice.append("Launch a targeted campaign or introduce bundle offers.")
        if "margin"     in f.lower(): advice.append("Audit top 3 cost drivers — 5% cost reduction doubles margin.")
        if "discount"   in f.lower(): advice.append("Cap discounts at 10% and use loyalty programs instead.")
        if "volatility" in f.lower(): advice.append("Introduce subscription or retainer revenue to stabilize sales.")

    return {"score": score, "grade": _grade(score), "metrics": metrics,
            "passed_checks": passed, "flags": flags, "advice": advice}


# ── HEALTH / SOCIAL MEDIA ─────────────────────
def score_health_social(df: pd.DataFrame, col: dict) -> dict:
    score, passed, flags, metrics = 0, [], [], {}

    sleep_col    = col.get("sleep")
    stress_col   = col.get("stress")
    anxiety_col  = col.get("anxiety")
    addiction_col = col.get("addiction")
    screen_col   = col.get("screen_time")
    social_col   = col.get("social_hours")
    physical_col = col.get("physical")
    academic_col = col.get("academic")
    depress_col  = col.get("depression")

    # Sleep quality
    if sleep_col and sleep_col in df.columns:
        avg_sleep = df[sleep_col].mean()
        metrics["avg_sleep_hours"] = round(float(avg_sleep), 2)
        if avg_sleep >= 7:   score += 20; passed.append("✅ Adequate average sleep (≥7 hrs)")
        elif avg_sleep >= 6: score += 10; passed.append("🟡 Below recommended sleep (6–7 hrs)")
        else:                flags.append("🚨 Critical sleep deprivation (<6 hrs average)")

    # Stress levels
    if stress_col and stress_col in df.columns:
        avg_stress = df[stress_col].mean()
        max_val    = df[stress_col].max()
        metrics["avg_stress_level"] = round(float(avg_stress), 2)
        pct_high   = (df[stress_col] > max_val * 0.6).mean() * 100
        metrics["high_stress_population_pct"] = round(float(pct_high), 1)
        if avg_stress < max_val * 0.40:   score += 20; passed.append("✅ Low average stress levels")
        elif avg_stress < max_val * 0.60: score += 10; passed.append("🟡 Moderate stress levels")
        else:                             flags.append("🚨 High stress levels detected in population")

    # Addiction levels
    if addiction_col and addiction_col in df.columns:
        avg_add = df[addiction_col].mean()
        max_val = df[addiction_col].max()
        metrics["avg_addiction_level"] = round(float(avg_add), 2)
        pct_high = (df[addiction_col] > max_val * 0.6).mean() * 100
        metrics["high_addiction_pct"] = round(float(pct_high), 1)
        if avg_add < max_val * 0.35:   score += 20; passed.append("✅ Low addiction levels")
        elif avg_add < max_val * 0.60: score += 10; passed.append("🟡 Moderate addiction levels")
        else:                          flags.append("🚨 High social media addiction detected")

    # Screen time
    if screen_col and screen_col in df.columns:
        avg_screen = df[screen_col].mean()
        metrics["avg_screen_time_before_sleep"] = round(float(avg_screen), 2)
        if avg_screen < 1:   score += 10; passed.append("✅ Low screen time before sleep")
        elif avg_screen < 2: score += 5
        else:                flags.append("⚠️ High screen time before sleep — impacts sleep quality")

    # Daily social media hours
    if social_col and social_col in df.columns:
        avg_social = df[social_col].mean()
        metrics["avg_daily_social_media_hours"] = round(float(avg_social), 2)
        if avg_social < 2:   score += 10; passed.append("✅ Healthy social media usage (<2 hrs/day)")
        elif avg_social < 4: score += 5;  passed.append("🟡 Moderate social media usage (2–4 hrs/day)")
        else:                flags.append("⚠️ Excessive social media usage (>4 hrs/day)")

    # Physical activity
    if physical_col and physical_col in df.columns:
        avg_phys = df[physical_col].mean()
        metrics["avg_physical_activity"] = round(float(avg_phys), 2)
        if avg_phys >= 1.0:   score += 10; passed.append("✅ Regular physical activity")
        else:                 flags.append("⚠️ Low physical activity levels")

    # Depression label
    if depress_col and depress_col in df.columns:
        dep_rate = df[depress_col].mean() * 100 if df[depress_col].dtype != object else \
                   (df[depress_col].astype(str).str.lower().isin(["1","yes","true","depressed"])).mean() * 100
        metrics["depression_rate_pct"] = round(float(dep_rate), 1)
        if dep_rate < 15:   score += 10; passed.append("✅ Low depression prevalence (<15%)")
        elif dep_rate < 30: score += 5;  passed.append("🟡 Moderate depression prevalence")
        else:               flags.append("🚨 High depression prevalence in dataset")

    score = min(score, 100)
    advice = []
    for f in flags:
        if "sleep"       in f.lower(): advice.append("Promote digital curfews — no screens 1hr before bed.")
        if "stress"      in f.lower(): advice.append("Introduce mindfulness or stress management programs.")
        if "addiction"   in f.lower(): advice.append("Consider app usage limits and scheduled device-free periods.")
        if "screen"      in f.lower(): advice.append("Encourage blue-light filters and bedtime mode features.")
        if "social media" in f.lower(): advice.append("Promote balanced social media usage with real-world interaction.")
        if "physical"    in f.lower(): advice.append("Encourage at least 30 mins of physical activity daily.")
        if "depression"  in f.lower(): advice.append("Recommend professional mental health support and community programs.")

    return {"score": score, "grade": _grade(score), "metrics": metrics,
            "passed_checks": passed, "flags": flags, "advice": advice}


# ── HR ────────────────────────────────────────
def score_hr(df: pd.DataFrame, col: dict) -> dict:
    score, passed, flags, metrics = 0, [], [], {}

    salary_col     = col.get("salary")
    attrition_col  = col.get("attrition")
    perf_col       = col.get("performance")
    dept_col       = col.get("department")

    if salary_col and salary_col in df.columns:
        metrics["avg_salary"]     = round(float(df[salary_col].mean()), 2)
        metrics["salary_std_dev"] = round(float(df[salary_col].std()), 2)
        cv = df[salary_col].std() / df[salary_col].mean() if df[salary_col].mean() else 0
        if cv < 0.30: score += 20; passed.append("✅ Balanced salary distribution")
        else:         flags.append("⚠️ High salary disparity across workforce")

    if attrition_col and attrition_col in df.columns:
        if df[attrition_col].dtype == object:
            rate = df[attrition_col].astype(str).str.lower().isin(["yes","1","true","left"]).mean() * 100
        else:
            rate = df[attrition_col].mean() * 100
        metrics["attrition_rate_pct"] = round(float(rate), 1)
        if rate < 10:   score += 30; passed.append("✅ Low attrition rate (<10%)")
        elif rate < 20: score += 15; passed.append("🟡 Moderate attrition (10–20%)")
        else:           flags.append("🚨 High attrition rate (>20%) — retention crisis")

    if perf_col and perf_col in df.columns:
        avg_perf = df[perf_col].mean()
        max_val  = df[perf_col].max()
        metrics["avg_performance_rating"] = round(float(avg_perf), 2)
        if avg_perf > max_val * 0.65: score += 30; passed.append("✅ High average performance rating")
        elif avg_perf > max_val * 0.5: score += 15; passed.append("🟡 Average performance is moderate")
        else:                          flags.append("⚠️ Low average performance rating")

    if dept_col and dept_col in df.columns:
        metrics["num_departments"] = int(df[dept_col].nunique())
        score += 5; passed.append("✅ Multi-department data available")

    score = min(score, 100)
    advice = []
    for f in flags:
        if "attrition" in f.lower(): advice.append("Implement stay interviews and career development programs.")
        if "salary"    in f.lower(): advice.append("Conduct pay equity audit to address compensation gaps.")
        if "performance" in f.lower(): advice.append("Introduce performance coaching and clearer KPI frameworks.")

    return {"score": score, "grade": _grade(score), "metrics": metrics,
            "passed_checks": passed, "flags": flags, "advice": advice}


# ── EDUCATION ─────────────────────────────────
def score_education(df: pd.DataFrame, col: dict) -> dict:
    score, passed, flags, metrics = 0, [], [], {}

    marks_col      = col.get("marks") or col.get("academic")
    attendance_col = col.get("attendance")
    subject_col    = col.get("subject")

    if marks_col and marks_col in df.columns:
        avg  = df[marks_col].mean()
        max_val = df[marks_col].max()
        metrics["avg_score"]       = round(float(avg), 2)
        metrics["pass_rate_pct"]   = round(float((df[marks_col] >= max_val * 0.4).mean() * 100), 1)
        if avg >= max_val * 0.65: score += 40; passed.append("✅ High average academic performance")
        elif avg >= max_val * 0.5: score += 20; passed.append("🟡 Moderate average performance")
        else:                     flags.append("🚨 Low average academic performance")

    if attendance_col and attendance_col in df.columns:
        avg_att = df[attendance_col].mean()
        max_val = df[attendance_col].max()
        metrics["avg_attendance_pct"] = round(float(avg_att / max_val * 100 if max_val > 1 else avg_att * 100), 1)
        if avg_att >= max_val * 0.75: score += 30; passed.append("✅ Good average attendance")
        else:                         flags.append("⚠️ Poor attendance — may impact results")

    score = min(score, 100)
    advice = []
    for f in flags:
        if "performance" in f.lower(): advice.append("Introduce tutoring programs and identify at-risk students early.")
        if "attendance"  in f.lower(): advice.append("Implement attendance tracking alerts for early intervention.")

    return {"score": score, "grade": _grade(score), "metrics": metrics,
            "passed_checks": passed, "flags": flags, "advice": advice}


# ── GENERAL (fallback for any unknown dataset) ─
def score_general(df: pd.DataFrame, col: dict) -> dict:
    score, passed, flags, metrics = 0, [], [], {}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(include="object").columns.tolist()

    metrics["total_rows"]    = len(df)
    metrics["total_columns"] = len(df.columns)
    metrics["numeric_cols"]  = len(numeric_cols)
    metrics["categorical_cols"] = len(cat_cols)

    # Completeness
    missing_pct = df.isnull().mean().mean() * 100
    metrics["missing_data_pct"] = round(float(missing_pct), 2)
    if missing_pct < 2:    score += 30; passed.append("✅ Excellent data completeness (<2% missing)")
    elif missing_pct < 10: score += 15; passed.append("🟡 Acceptable completeness (<10% missing)")
    else:                  flags.append("⚠️ High missing data — quality issues detected")

    # Duplicates
    dup_pct = (df.duplicated().sum() / len(df)) * 100
    metrics["duplicate_rows_pct"] = round(float(dup_pct), 2)
    if dup_pct < 1:  score += 20; passed.append("✅ Very few duplicate rows (<1%)")
    elif dup_pct < 5: score += 10
    else:            flags.append("⚠️ Many duplicate rows detected")

    # Numeric diversity
    if numeric_cols:
        avg_cv = pd.Series([
            df[c].std() / df[c].mean() for c in numeric_cols if df[c].mean() != 0
        ]).mean()
        metrics["avg_numeric_variation"] = round(float(avg_cv), 4)
        if avg_cv > 0.1: score += 20; passed.append("✅ Good numeric variation in dataset")

    # Size bonus
    if len(df) >= 1000:  score += 15; passed.append("✅ Large dataset (≥1000 rows)")
    elif len(df) >= 100: score += 10; passed.append("🟡 Medium dataset (100–999 rows)")
    else:                flags.append("⚠️ Small dataset — insights may not be statistically significant")

    # Column richness
    if len(df.columns) >= 8: score += 10; passed.append("✅ Rich feature set (≥8 columns)")

    score = min(score, 100)
    advice = []
    for f in flags:
        if "missing"    in f.lower(): advice.append("Use imputation or flag rows with missing values before analysis.")
        if "duplicate"  in f.lower(): advice.append("Deduplicate records to avoid skewed metrics.")
        if "small"      in f.lower(): advice.append("Collect more data points for statistically reliable insights.")

    return {"score": score, "grade": _grade(score), "metrics": metrics,
            "passed_checks": passed, "flags": flags, "advice": advice}


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

DOMAIN_LABELS = {
    "sales":        "📦 Sales & Revenue",
    "ecommerce":    "🛒 E-Commerce",
    "finance":      "💰 Finance",
    "health":       "🏥 Health & Wellness",
    "social_media": "📱 Social Media & Mental Health",
    "hr":           "👥 Human Resources",
    "education":    "🎓 Education",
    "general":      "📊 General Dataset",
}

def financial_scorecard(df: pd.DataFrame) -> dict:
    """
    Universal scorecard — auto-detects domain and scores accordingly.
    Never raises — always returns a dict.
    """
    try:
        # Step 1: detect domain
        domain = detect_domain(df)

        # Step 2: fuzzy-resolve columns
        col_map = fuzzy_resolve_columns(df)
        print(f"[Scorecard] Resolved columns: {col_map}")

        # Step 3: run domain-specific scorer
        scorers = {
            "sales":        score_sales,
            "ecommerce":    score_sales,       # reuse sales scorer
            "finance":      score_sales,       # reuse sales scorer
            "health":       score_health_social,
            "social_media": score_health_social,
            "hr":           score_hr,
            "education":    score_education,
            "general":      score_general,
        }
        scorer = scorers.get(domain, score_general)
        result = scorer(df, col_map)

        return {
            **result,
            "domain":          domain,
            "domain_label":    DOMAIN_LABELS.get(domain, "📊 General Dataset"),
            "column_mapping":  col_map,
            "dataset_shape":   {"rows": len(df), "columns": len(df.columns)},
        }

    except Exception as e:
        print(f"[Scorecard Error] {e}")
        return {"error": f"Scorecard failed: {str(e)}"}