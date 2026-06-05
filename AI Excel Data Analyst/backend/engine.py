import pandas as pd
import plotly.express as px
import plotly
import google.generativeai as genai
import json
import re
import os
import duckdb
import threading
import builtins

# ─────────────────────────────────────────────
# 🔑  GEMINI SETUP
# ─────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyBHguB0sSCvFb9D1KqHkzcRZtoM6IcwyCo"))

_model = genai.GenerativeModel("gemini-2.5-flash")


# ─────────────────────────────────────────────
# 🧹  1. DATA CLEANING ENGINE
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df


# ─────────────────────────────────────────────
# 📊  2. DATA PROFILER MODULE
# ─────────────────────────────────────────────
def profile_data(df: pd.DataFrame) -> dict:
    return {
        "rows":           len(df),
        "columns":        len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types":     df.dtypes.astype(str).to_dict(),
    }


# ─────────────────────────────────────────────
# 📈  3. KPI ENGINE
# ─────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    kpis = {}
    for col in df.select_dtypes(include="number").columns:
        kpis[col] = {
            "sum":  float(df[col].sum())  if pd.notnull(df[col].sum())  else 0.0,
            "mean": float(df[col].mean()) if pd.notnull(df[col].mean()) else 0.0,
            "max":  float(df[col].max())  if pd.notnull(df[col].max())  else 0.0,
            "min":  float(df[col].min())  if pd.notnull(df[col].min())  else 0.0,
        }
    return kpis


# ─────────────────────────────────────────────
# 🔍  4. ANOMALY DETECTION MODULE
# ─────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame) -> dict:
    anomalies = {}
    for col in df.select_dtypes(include="number").columns:
        mean = df[col].mean()
        std  = df[col].std()
        mask = (df[col] > mean + 2 * std) | (df[col] < mean - 2 * std)
        anomaly_df = df.loc[mask, col]
        if len(anomaly_df) > 0:
            anomalies[col] = {str(k): float(v) for k, v in anomaly_df.to_dict().items()}
    return anomalies


# ─────────────────────────────────────────────
# 🧠  5. INTENT DETECTION
# ─────────────────────────────────────────────
def detect_intent(query: str) -> str:
    prompt = f"""Classify this query into exactly ONE of these categories:
trend, comparison, ranking, anomaly, summary, prediction

Query: {query}

Return only the single lowercase word — nothing else. No punctuation."""
    try:
        res  = _model.generate_content(prompt)
        word = res.text.strip().lower().strip('`').split()[0]
        return word
    except Exception:
        return "summary"


# ─────────────────────────────────────────────
# 🧾  6. JSON EXTRACTOR  (handles Gemini fences)
# ─────────────────────────────────────────────
def extract_json(text: str) -> dict | None:
    if not text:
        return None

    # Strategy 1: ```json ... ``` or ``` ... ``` fences
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    # Strategy 2: outermost { ... } by brace depth
    try:
        start = text.index('{')
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    except Exception:
        pass

    # Strategy 3: strip all fences, parse whole text
    try:
        clean = re.sub(r'```(?:json)?', '', text).strip().rstrip('`')
        return json.loads(clean)
    except Exception:
        pass

    print(f"[extract_json] FAILED — raw output:\n{text[:600]}")
    return None


# ─────────────────────────────────────────────
# ⚡  7. GEMINI LLM CALL  (+ cache)
# ─────────────────────────────────────────────
_cache: dict = {}

def call_llm(prompt: str) -> dict | None:
    if prompt in _cache:
        return _cache[prompt]
    try:
        res    = _model.generate_content(prompt)
        raw    = res.text
        print(f"[LLM Raw Response]:\n{raw[:1000]}")
        parsed = extract_json(raw)
        if parsed:
            _cache[prompt] = parsed
        return parsed
    except Exception as e:
        print(f"[LLM Error] {e}")
        return None


# ─────────────────────────────────────────────
# 🧠  8. PROMPT ENGINE
# ─────────────────────────────────────────────
def build_prompt(query: str, df: pd.DataFrame, intent: str) -> str:
    return f"""You are a senior data analyst AI assistant.

INTENT: {intent}

STRICT RULES:
- Use ONLY the dataframe variable called `df`
- Do NOT add any import statements — pandas is already available as `pd`
- Store the final answer in a variable named exactly: result
- `result` MUST be a pandas DataFrame or Series (NOT a scalar value)
- If result would be a single number, wrap it: result = pd.DataFrame({{"value": [the_number]}})
- After any groupby, call .reset_index() so group keys become regular columns
- Example: result = df.groupby("col")["val"].sum().reset_index()
- YOUR RESPONSE MUST BE RAW JSON ONLY — absolutely no markdown, no triple backticks

AVAILABLE COLUMNS:
{list(df.columns)}

SAMPLE DATA (first 5 rows):
{df.head(5).astype(str).to_dict()}

USER QUERY:
{query}

Respond with this EXACT raw JSON (no ```json wrapper, no extra text):
{{
  "code": "result = df.groupby('col')['val'].sum().reset_index()",
  "chart": "bar",
  "insight": "plain English insight here",
  "follow_up": "a suggested follow-up question"
}}

chart value must be exactly one of: bar, line, pie, none"""


# ─────────────────────────────────────────────
# 🚨  9. SAFE CODE EXECUTION ENGINE
# ─────────────────────────────────────────────
def safe_execute(code: str, df: pd.DataFrame, timeout_seconds: int = 15):
    """
    FIX: Pass real __builtins__ so built-in functions (len, range, sorted,
    list, dict, print, etc.) work inside exec'd code.
    Dangerous tokens are blocked by keyword scan before execution.
    """
    BANNED = [
        "import os", "import sys", "subprocess",
        "open(", "exec(", "eval(", "__import__",
        "shutil", "socket",
    ]
    for token in BANNED:
        if token in code:
            raise ValueError(f"Unsafe code blocked — contains: '{token}'")

    # ✅ Use real builtins so len/range/sorted/etc. all work
    safe_globals = {
        "__builtins__": builtins,
        "pd":           pd,
    }
    local_vars  = {"df": df.copy(), "pd": pd}
    exec_errors: list = []

    def _run():
        try:
            exec(code, safe_globals, local_vars)
        except Exception as e:
            exec_errors.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Code execution exceeded {timeout_seconds}s — killed.")

    if exec_errors:
        raise exec_errors[0]

    return local_vars.get("result", None)


# ─────────────────────────────────────────────
# 📊  10. SMART CHART ENGINE
# ─────────────────────────────────────────────
def chart_engine(df: pd.DataFrame, chart_type: str) -> dict | None:
    if df is None or len(df) == 0:
        return None

    # Always reset index so groupby keys become x-axis columns
    df = df.reset_index(drop=False)

    # Normalize chart_type — handle whatever Gemini might return
    chart_type = str(chart_type).lower().strip()
    if   "bar"  in chart_type: chart_type = "bar"
    elif "line" in chart_type: chart_type = "line"
    elif "pie"  in chart_type: chart_type = "pie"
    else:                       chart_type = "none"

    # Separate label columns (object/datetime) from numeric columns
    label_cols   = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("datetime")]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    x_col = label_cols[0]   if label_cols   else df.columns[0]
    y_col = numeric_cols[0] if numeric_cols else df.columns[-1]

    print(f"[Chart Engine] type={chart_type}  x={x_col}  y={y_col}  shape={df.shape}")

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=numeric_cols if numeric_cols else [y_col],
                         barmode="group")

        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=numeric_cols if numeric_cols else [y_col],
                          markers=True)

        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col)

        else:
            return None

        fig.update_layout(
            title="AI Insight Chart",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, l=20, r=20, b=20),
        )
        return json.loads(json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder))

    except Exception as e:
        print(f"[Chart Engine Error] {e}")
        return None


# ─────────────────────────────────────────────
# 🚀  11. MAIN AI PIPELINE
# ─────────────────────────────────────────────
def run_query_engine(query: str, df: pd.DataFrame) -> dict:
    intent = detect_intent(query)
    print(f"[Intent] {intent}")

    prompt   = build_prompt(query, df, intent)
    response = call_llm(prompt)

    if not response:
        return {"error": "AI could not generate a valid response. Check uvicorn terminal for raw LLM output."}

    insight    = response.get("insight",   "No insight available.")
    follow_up  = response.get("follow_up", "")
    chart_type = response.get("chart",     "none")
    code       = response.get("code",      "")

    print(f"[Generated Code]:\n{code}")

    if not code:
        return {
            "intent": intent, "insight": insight, "follow_up": follow_up,
            "chart_json": None, "result_data": None, "code": "",
        }

    result_df_dict = None
    chart_json     = None

    try:
        result = safe_execute(code, df)

        if isinstance(result, pd.Series):
            result = result.reset_index()          # Series → DataFrame with index as column

        if isinstance(result, pd.DataFrame) and len(result) > 0:
            result_df_dict = result.head(100).astype(str).to_dict(orient="records")
            chart_json     = chart_engine(result.copy(), chart_type)

    except TimeoutError as te:
        return {"error": str(te), "insight": insight, "intent": intent}
    except Exception as e:
        return {"error": f"Code Execution Error: {str(e)}", "insight": insight, "intent": intent}

    return {
        "intent":      intent,
        "insight":     insight,
        "follow_up":   follow_up,
        "chart_json":  chart_json,
        "result_data": result_df_dict,
        "code":        code,
    }


# ─────────────────────────────────────────────
# 🛠️  UTILITY HELPERS
# ─────────────────────────────────────────────
def load_dataframe(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")
    return clean_data(df)


def get_full_analysis(df: pd.DataFrame) -> dict:
    return {
        "profile":   profile_data(df),
        "kpis":      compute_kpis(df),
        "anomalies": detect_anomalies(df),
    }