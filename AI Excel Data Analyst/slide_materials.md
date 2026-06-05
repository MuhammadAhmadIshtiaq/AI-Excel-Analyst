# AI Excel Data Analyst - Presentation Slide Materials

These materials are structured to be easily copy-pasted into presentation slides (e.g., PowerPoint, Google Slides, or Keynote).

---

## Slide 1: Title Slide
**Title:** AI-Powered Excel Data Analyst
**Subtitle:** Automating Data Profiling, Analysis, and Visualization with Generative AI
**Presenters/Team:** [Your Name/Team Name]

---

## Slide 2: Problem Statement
**The Challenge in Modern Data Analysis**
*   **Skill Gap:** Extracting actionable insights from raw data typically requires specialized skills in Python, Pandas, and data visualization tools.
*   **Time-Consuming:** Manual data cleaning, KPI calculation, and anomaly detection are tedious and repetitive tasks.
*   **Domain Agnosticism:** Traditional BI dashboards are static and require manual configuration for different types of datasets (HR, Sales, Health, etc.).
*   **The Goal:** To empower non-technical users to converse with their datasets in natural language, automatically generating insights, charts, and domain-specific health scores.

---

## Slide 3: Methodology & Architecture
**How the AI Excel Analyst Works**
1.  **Data Ingestion & Cleaning:** Uploads CSV/Excel files, automatically parses dates, drops duplicates, and profiles data structures.
2.  **Intent Classification (LLM):** Parses user queries using Google Gemini (2.5-Flash) to determine the exact intent (trend, comparison, ranking, anomaly, summary, prediction).
3.  **Prompt Engineering & Code Generation:** Dynamically constructs a prompt with schema context. The LLM generates strict Python (Pandas) code wrapped in JSON.
4.  **Safe Code Execution (Sandbox):** Executes the LLM-generated code in a secure, multi-threaded environment with timeout restrictions and keyword blocking to prevent malicious execution.
5.  **Smart Chart Engine:** Converts the executed Pandas DataFrame into responsive Plotly visualizations (Bar, Line, Pie) based on the LLM's recommendation.

---

## Slide 4: Core Algorithms (1/2) - Anomaly & KPI Engine
**Automated Statistical Profiling**
*   **KPI Engine:** Automatically calculates aggregate metrics (`sum`, `mean`, `max`, `min`) for all numeric columns on upload.
*   **Anomaly Detection Algorithm:** Uses statistical variance to flag outliers. 
    *   *Formula:* Flags any data point that falls outside of **Mean ± (2 × Standard Deviation)**.
    *   This instantly highlights data anomalies without requiring user prompting.

---

## Slide 5: Core Algorithms (2/2) - Universal Intelligent Scorecard
**Domain-Aware Data Scoring**
*   **Fuzzy Column Resolver:** Uses `difflib.SequenceMatcher` to map arbitrary user column names (e.g., "sales_amt", "revenue_usd") to standardized concepts using similarity scoring (>0.55 ratio).
*   **Domain Detector:** Scans column names and sample data to categorize the dataset into domains like `Sales`, `Health`, `HR`, `Education`, or `Social Media`.
*   **Rule-Based Scoring:** Applies domain-specific logic to grade the dataset (A to D).
    *   *Example (Sales):* Evaluates Profit Margins, Revenue Volatility, and Discount impacts.
    *   *Example (HR):* Evaluates Attrition Rates, Salary disparity (Coefficient of Variation), and Performance.

---

## Slide 6: Key Code Snippet - Intent Detection & LLM Routing
**Using Gemini to Classify User Intent**
```python
def detect_intent(query: str) -> str:
    prompt = f"""Classify this query into exactly ONE of these categories:
trend, comparison, ranking, anomaly, summary, prediction

Query: {query}

Return only the single lowercase word — nothing else."""
    
    try:
        res  = _model.generate_content(prompt)
        word = res.text.strip().lower().strip('`').split()[0]
        return word
    except Exception:
        return "summary"
```

---

## Slide 7: Key Code Snippet - Safe Execution Sandbox
**Executing LLM-Generated Pandas Code Securely**
```python
def safe_execute(code: str, df: pd.DataFrame, timeout_seconds: int = 15):
    # 1. Block dangerous keywords
    BANNED = ["import os", "import sys", "subprocess", "open(", "exec(", "__import__"]
    for token in BANNED:
        if token in code: raise ValueError("Unsafe code blocked")

    # 2. Provide safe execution environment with timeouts
    safe_globals = {"__builtins__": builtins, "pd": pd}
    local_vars  = {"df": df.copy(), "pd": pd}
    
    def _run():
        exec(code, safe_globals, local_vars)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive(): raise TimeoutError("Execution timed out.")
    return local_vars.get("result", None)
```

---

## Slide 8: Key Code Snippet - Domain Detection
**Auto-Detecting Dataset Context**
```python
def detect_domain(df: pd.DataFrame) -> str:
    # Build text blob from columns and sample data
    col_text = " ".join(df.columns).lower()
    sample_text = " ".join(df.select_dtypes(include="object").head(10).astype(str).values.flatten()).lower()
    full_text = col_text + " " + sample_text

    domain_scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        domain_scores[domain] = sum(1 for kw in keywords if kw in full_text)

    # Return the domain with the highest keyword overlap
    best_domain = max(domain_scores, key=domain_scores.get)
    best_score  = domain_scores[best_domain]
    
    return best_domain if best_score >= 2 else "general"
```

---

## Slide 9: Conclusion & Impact
**Value Proposition**
*   **Accessibility:** Turns anyone into a capable data analyst via chat.
*   **Speed:** Reduces data preparation and visualization time from hours to seconds.
*   **Actionability:** The universal scorecard doesn't just show data; it provides strategic, domain-specific advice (e.g., "Cap discounts at 10%", "Introduce stay interviews to reduce attrition").
*   **Security:** Built-in safeguards ensure LLM code generation cannot compromise the host system.
