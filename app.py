from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).parent
SAMPLE_CSV = APP_DIR / "data" / "campus_event_registrations.csv"
st.set_page_config(page_title="Ask Your Data | AI Analytics Portal", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1.2rem;padding-bottom:2.5rem}
.hero {padding:1.35rem 1.5rem;border-radius:18px;background:linear-gradient(120deg,#10233f,#17609a);color:#fff;margin-bottom:1rem}
.hero h1 {color:#fff;margin:0}.hero p {color:#e8f2fc;margin:.4rem 0 0}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.28);padding:12px;border-radius:12px}
</style>
<div class="hero"><h1>📊 Ask Your Data</h1>
<p>Upload a dataset, ask questions in plain English, and explore validated analytics with tables, charts, and summaries.</p></div>
""", unsafe_allow_html=True)

# ---------------------------- Data handling ----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    names = []
    used = set()
    for i, col in enumerate(result.columns):
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(col).strip().lower()).strip("_")
        name = name or f"column_{i+1}"
        base, n = name, 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        names.append(name)
    result.columns = names
    return result

def smart_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        if pd.api.types.is_object_dtype(result[col]) or pd.api.types.is_string_dtype(result[col]):
            name_hint = any(word in col.lower() for word in ("date", "time", "timestamp", "month", "year"))
            sample = result[col].dropna().astype(str).head(30)
            if name_hint and len(sample):
                parsed = pd.to_datetime(result[col], errors="coerce")
                if parsed.notna().mean() >= 0.5:
                    result[col] = parsed
    return result

def load_any_file(file) -> pd.DataFrame:
    suffix = Path(file.name).suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Please upload a CSV, XLSX, or XLS file.")
    if df.empty:
        raise ValueError("The uploaded dataset has no rows.")
    df = normalize_columns(df)
    df = smart_parse_dates(df)
    # Convert numeric-looking strings where doing so is safe and useful.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            series = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            converted = pd.to_numeric(series, errors="coerce")
            nonempty = df[col].notna().sum()
            if nonempty and converted.notna().sum() / nonempty >= 0.9:
                df[col] = converted
    return df

def column_groups(df):
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric and c not in dates]
    return numeric, categorical, dates

def schema_summary(df):
    num, cat, dates = column_groups(df)
    return {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "numeric": num, "categorical": cat, "dates": dates,
        "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum())
    }

# ---------------------------- Question engine ----------------------------
def find_column_in_question(question: str, columns: list[str]) -> str | None:
    q = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    # Prefer exact column names and common variants, longest first.
    candidates = sorted(columns, key=len, reverse=True)
    for col in candidates:
        variants = {col.lower(), col.lower().replace("_", " "), col.lower().replace("_", "")}
        for variant in variants:
            if variant and re.search(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", q):
                return col
    # Match distinctive parts of a column name.
    for col in candidates:
        parts = [p for p in col.lower().split("_") if len(p) > 2]
        if parts and all(p in q.split() for p in parts):
            return col
    return None

def find_filter(df: pd.DataFrame, question: str):
    q = question.lower()
    # Find a unique text value explicitly mentioned in the question.
    possible = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = df[col].dropna().astype(str).unique()
        for value in vals:
            value_clean = str(value).strip()
            if len(value_clean) >= 2 and value_clean.lower() in q:
                possible.append((len(value_clean), col, value_clean))
    if not possible:
        return None, None
    _, col, val = sorted(possible, reverse=True)[0]
    return col, val

def infer_operation(question: str, df: pd.DataFrame) -> dict[str, Any]:
    """
    Translate common natural-language analytics questions into a small, validated plan.
    Grouping columns are parsed separately from metric columns to avoid accidentally
    grouping by the numeric measure (a common source of wrong answers).
    """
    q = re.sub(r"[^a-z0-9_ ]+", " ", question.lower())
    q = re.sub(r"\s+", " ", q).strip()
    numeric, categorical, dates = column_groups(df)
    columns = list(df.columns)

    def match_col(text: str, candidates: list[str]) -> str | None:
        text = re.sub(r"[^a-z0-9_ ]+", " ", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        for c in sorted(candidates, key=len, reverse=True):
            variants = {c.lower(), c.lower().replace("_", " "), c.lower().replace("_", "")}
            if any(re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", text) for v in variants if v):
                return c
        return None

    # Identify an explicit grouping dimension from text after "by", "per", or "wise".
    group = None
    group_match = re.search(r"\b(?:group(?:ed)?\s+by|by|per)\s+(.+?)(?:\s+(?:and|with|where|for|from|in|over|sorted|ordered)\b|$)", q)
    if group_match:
        group_text = group_match.group(1).strip()
        group = match_col(group_text, columns)
        if group is None:
            # "by month/day/year" can refer to a date field.
            if any(x in group_text for x in ("month", "monthly", "year", "yearly", "day", "daily", "date", "time")):
                group = dates[0] if dates else None
    if group is None:
        wise_match = re.search(r"\b([a-z0-9_ ]+?)\s*[- ]wise\b", q)
        if wise_match:
            group = match_col(wise_match.group(1), columns)
    if group is None and any(x in q for x in ("monthly trend", "monthly registrations", "by month", "over time", "trend over time", "daily trend", "yearly trend")):
        group = dates[0] if dates else None
    if group is None and any(x in q for x in ("distribution", "breakdown", "category counts")):
        group = match_col(q, categorical) or (categorical[0] if categorical else None)

    # Identify metric column independently of the grouping dimension.
    col = match_col(q, columns)
    if col == group and col not in numeric:
        # A category mentioned as the grouping dimension is not the numeric metric.
        col = match_col(q, numeric)
    if col not in numeric and any(x in q for x in ("average", "avg", "mean", "median", "sum", "total", "maximum", "max", "minimum", "min", "highest", "lowest", "largest", "smallest", "correlation")):
        col = match_col(q, numeric)

    # Operations: prioritize explicit numeric aggregate words before generic "top/highest".
    if re.search(r"\b(average|avg|mean)\b", q):
        op = "mean"
    elif re.search(r"\bmedian\b", q):
        op = "median"
    elif re.search(r"\b(sum|summed|total)\b", q) and not re.search(r"\b(total|how many)\s+(?:number of\s+)?(?:rows|records|registrations|entries|orders|users)\b", q):
        op = "sum"
    elif re.search(r"\b(missing|null|blank|empty values|data quality)\b", q):
        op = "missing"
    elif re.search(r"\b(unique|distinct|different)\b", q):
        op = "nunique"
    elif re.search(r"\b(correlation|correlate|relationship between)\b", q):
        op = "correlation"
    elif re.search(r"\b(minimum|lowest|smallest|min)\b", q):
        op = "min" if col in numeric else "count"
    elif re.search(r"\b(maximum|highest|largest|max)\b", q):
        op = "max" if col in numeric and not group else "count"
    elif re.search(r"\b(top|most|highest|largest)\s+\d+\b", q):
        op = "count"
    else:
        op = "count"

    # "Which department has the highest feedback score?" means aggregate score by department,
    # then identify the department with the largest aggregate.
    if re.search(r"\b(which|what)\s+.+\s+(has|have|with)\s+the\s+(highest|largest|lowest|smallest)\b", q):
        group_candidate = match_col(q, categorical)
        metric_candidate = match_col(q, numeric)
        if group_candidate:
            group = group_candidate
        if metric_candidate:
            col = metric_candidate
            op = "mean"

    # Explicit top-N questions usually request a ranked grouping.
    top_match = re.search(r"\btop\s+(\d{1,3})\b", q)
    limit = min(max(int(top_match.group(1)), 1), 100) if top_match else 25

    # If a date grouping was requested, the date column is the dimension, not a metric.
    if group in dates and col == group:
        col = match_col(q, numeric)

    # For count requests, the target column is optional; count means number of rows.
    if op in {"mean", "median", "sum", "max", "min"} and col not in numeric:
        col = numeric[0] if numeric else None

    if op == "correlation":
        mentioned_numeric = []
        for c in sorted(numeric, key=len, reverse=True):
            if match_col(q, [c]) and c not in mentioned_numeric:
                mentioned_numeric.append(c)
        second = mentioned_numeric[1] if len(mentioned_numeric) > 1 else (numeric[1] if len(numeric) > 1 else None)
        col = mentioned_numeric[0] if mentioned_numeric else (numeric[0] if numeric else None)
    else:
        second = None

    filter_col, filter_val = find_filter(df, question)
    # Do not treat a value as a filter if it is only the name of a group/metric column.
    if filter_col in {group, col} and filter_val is None:
        filter_col, filter_val = None, None

    return {
        "operation": op, "column": col, "group_by": group,
        "filter_column": filter_col, "filter_value": filter_val,
        "second_column": second, "limit": limit
    }

def validate_operation(plan, df):
    allowed_ops = {"count", "mean", "median", "sum", "max", "min", "nunique", "missing", "correlation"}
    if plan["operation"] not in allowed_ops:
        raise ValueError("Unsupported operation.")
    if plan["column"] is not None and plan["column"] not in df.columns:
        plan["column"] = None
    if plan["group_by"] is not None and plan["group_by"] not in df.columns:
        plan["group_by"] = None
    if plan["filter_column"] is not None:
        if plan["filter_column"] not in df.columns:
            plan["filter_column"], plan["filter_value"] = None, None
        elif plan["filter_value"] not in set(df[plan["filter_column"]].dropna().astype(str)):
            plan["filter_column"], plan["filter_value"] = None, None
    if plan["operation"] in {"mean", "median", "sum", "max", "min", "correlation"}:
        if plan["column"] is None or not pd.api.types.is_numeric_dtype(df[plan["column"]]):
            # Dates can be counted but should not be used for numeric aggregates.
            plan["operation"] = "count"
            plan["column"] = None
    if plan["operation"] == "correlation":
        second = plan.get("second_column")
        if second not in df.columns or not pd.api.types.is_numeric_dtype(df[second]):
            plan["operation"] = "count"
            plan["column"] = None
    plan["limit"] = max(1, min(int(plan.get("limit", 25)), 100))
    return plan

def execute_plan(df: pd.DataFrame, plan: dict[str, Any]):
    work = df.copy()
    if plan["filter_column"] and plan["filter_value"] is not None:
        work = work[work[plan["filter_column"]].astype(str) == str(plan["filter_value"])]
    op, col, group = plan["operation"], plan["column"], plan["group_by"]
    if op == "correlation":
        result = pd.DataFrame([{"column_1": col, "column_2": plan["second_column"],
                                "correlation": work[col].corr(work[plan["second_column"]]) if len(work) > 1 else None}])
        summary = "Correlation measures the linear relationship between two numeric columns; values closer to -1 or 1 indicate stronger linear relationships."
        return work, result, summary
    if op == "missing":
        result = pd.DataFrame({"column": work.columns, "missing_values": [int(work[c].isna().sum()) for c in work.columns]})
        result = result.sort_values("missing_values", ascending=False).head(plan["limit"])
        return work, result, f"Missing values were counted for each column across {len(work)} matching rows."
    if group:
        if group in column_groups(work)[2]:
            # Aggregate date values at a sensible time grain.
            grain = "M" if any(x in st.session_state.get("current_question", "").lower() for x in ["monthly", "month", "yearly", "year"]) else "D"
            work = work.copy()
            work["_group_label"] = pd.to_datetime(work[group], errors="coerce").dt.to_period(grain).astype(str)
            group_col = "_group_label"
            display_group = group if grain == "D" else "month"
        else:
            group_col = group
            display_group = group
        if op == "count":
            result = work.groupby(group_col, dropna=False).size().reset_index(name="record_count")
            value_col = "record_count"
        elif op == "nunique":
            target = col or group
            result = work.groupby(group_col, dropna=False)[target].nunique().reset_index(name=f"unique_{target}")
            value_col = f"unique_{target}"
        else:
            target = col
            if target is None:
                result = work.groupby(group_col, dropna=False).size().reset_index(name="record_count")
                value_col = "record_count"
            else:
                result = work.groupby(group_col, dropna=False)[target].agg(op).reset_index(name=f"{op}_{target}")
                value_col = f"{op}_{target}"
        if group_col == "_group_label":
            result = result.rename(columns={"_group_label": display_group})
        if group in column_groups(df)[2]:
            result = result.sort_values(display_group, ascending=True).head(plan["limit"]).reset_index(drop=True)
        else:
            result = result.sort_values(value_col, ascending=False).head(plan["limit"]).reset_index(drop=True)
        summary = f"Calculated {op} grouped by {display_group} using {len(work)} matching records. The table and chart show {len(result)} group(s)."
        return work, result, summary
    if op == "count":
        result = pd.DataFrame([{"record_count": int(len(work))}])
        summary = f"There are {len(work):,} matching records."
    elif op == "nunique":
        target = col or (column_groups(work)[1][0] if column_groups(work)[1] else work.columns[0])
        result = pd.DataFrame([{f"unique_{target}": int(work[target].nunique(dropna=True))}])
        summary = f"Column '{target}' contains {work[target].nunique(dropna=True):,} distinct non-missing values."
    elif op == "missing":
        result = pd.DataFrame({"column": work.columns, "missing_values": [int(work[c].isna().sum()) for c in work.columns]})
        summary = "Missing values were counted for each column."
    else:
        target = col
        value = getattr(work[target], op)() if len(work) else None
        result = pd.DataFrame([{f"{op}_{target}": value}])
        summary = f"The {op} of '{target}' is {value}." if pd.notna(value) else f"No valid values were available for '{target}'."
    return work, result, summary

def auto_chart(result: pd.DataFrame, question: str = ""):
    """Always create a Plotly chart for a non-empty result."""
    if result is None or result.empty:
        return None
    frame = result.copy()
    # Date/time labels should remain readable and ordered.
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    categorical = [c for c in frame.columns if c not in numeric]
    if numeric and categorical:
        x, y = categorical[0], numeric[0]
        if "correlation" in y.lower():
            return px.bar(frame, x=x, y=y, title="Correlation result")
        if len(frame) > 18:
            return px.line(frame, x=x, y=y, markers=True, title=f"{y.replace('_',' ').title()} by {x.replace('_',' ').title()}")
        return px.bar(frame, x=x, y=y, text_auto=".3s", title=f"{y.replace('_',' ').title()} by {x.replace('_',' ').title()}")
    if len(numeric) >= 2:
        return px.scatter(frame, x=numeric[0], y=numeric[1], title="Numeric relationship")
    if numeric:
        label = "Metric"
        return px.bar(frame, x=[label] * len(frame), y=numeric[0], text_auto=".3s", title=question[:60] or "Analytics result")
    # If results are nonnumeric, chart their frequency.
    first = frame.columns[0]
    counts = frame[first].astype(str).value_counts().head(20).rename_axis("value").reset_index(name="count")
    return px.bar(counts, x="value", y="count", title=f"Distribution of {first.replace('_',' ')}")

def llm_answer(question: str, df: pd.DataFrame, api_key: str, model: str):
    """Optional OpenAI-compatible interpretation. Never executes model-generated code."""
    from openai import OpenAI
    numeric, categorical, dates = column_groups(df)
    schema = {c: {"dtype": str(df[c].dtype), "sample_values": df[c].dropna().astype(str).head(5).tolist()} for c in df.columns}
    prompt = f"""You are a data analyst. Return JSON only with keys operation, column, group_by, filter_column, filter_value, second_column, limit.
Allowed operations: count, mean, median, sum, max, min, nunique, missing, correlation.
Use only exact column names from this schema. column and group_by must be schema column names or null. For mean/median/sum/max/min use numeric columns only. filter_value must be a value shown in sample_values or null. Do not write SQL, Python, or code.
Schema: {json.dumps(schema, default=str)}
Question: {question}"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=[{"role":"system","content":"Translate questions into a constrained analytics plan. Return valid JSON only."},
                  {"role":"user","content":prompt}],
        response_format={"type":"json_object"}, temperature=0
    )
    plan = json.loads(response.choices[0].message.content)
    plan.setdefault("operation", "count")
    plan.setdefault("column", None)
    plan.setdefault("group_by", None)
    plan.setdefault("filter_column", None)
    plan.setdefault("filter_value", None)
    plan.setdefault("second_column", None)
    plan.setdefault("limit", 25)
    return plan

# ---------------------------- UI ----------------------------
if "df" not in st.session_state:
    try:
        st.session_state.df = load_any_file(open(SAMPLE_CSV, "rb"))
        st.session_state.filename = SAMPLE_CSV.name
    except Exception:
        st.session_state.df = pd.DataFrame()
        st.session_state.filename = "No dataset loaded"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""

with st.sidebar:
    st.header("⚙️ Dataset & settings")
    upload = st.file_uploader("Upload CSV / Excel dataset", type=["csv", "xlsx", "xls"])
    if upload is not None and upload.name != st.session_state.filename:
        try:
            st.session_state.df = load_any_file(upload)
            st.session_state.filename = upload.name
            st.session_state.last_result = None
            st.success(f"Loaded {upload.name}")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load file: {exc}")
    if st.button("Restore sample dataset", use_container_width=True):
        st.session_state.df = load_any_file(open(SAMPLE_CSV, "rb"))
        st.session_state.filename = SAMPLE_CSV.name
        st.session_state.last_result = None
        st.rerun()
    st.divider()
    api_key = st.text_input("OpenAI API key (optional)", value=os.getenv("OPENAI_API_KEY", ""), type="password", help="Only needed for broader LLM interpretation. Do not share your key.")
    model = st.text_input("LLM model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    use_llm = st.checkbox("Use LLM for question interpretation", value=bool(api_key))
    st.caption("LLM: API key detected" if api_key else "LLM: API key not configured")
    if st.button("Reset conversation", use_container_width=False):
        st.session_state.last_result = None
        st.session_state.last_plan = None
        st.session_state.last_question = ""
        st.session_state.last_summary = ""
        st.session_state.pop("sql_last_result", None)
        st.rerun()
    st.caption("Without an API key, the app uses a schema-aware local question engine.")

df = st.session_state.df
if df.empty:
    st.error("No data loaded. Upload a CSV or Excel file using the sidebar.")
    st.stop()

numeric_cols, categorical_cols, date_cols = column_groups(df)
profile = schema_summary(df)

with st.sidebar:
    page = st.radio(
        "Navigate",
        ["Ask Your Data", "Dashboard", "Dataset Explorer", "SQL Explorer", "About"],
        index=0,
        key="main_navigation",
    )

if page == "Ask Your Data":
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", len(df.columns))
    m3.metric("Missing cells", f"{profile['missing']:,}")
    m4.metric("Duplicate rows", f"{profile['duplicates']:,}")
    st.caption(f"Loaded: **{st.session_state.filename}**")
    with st.expander("Suggested questions (generated from your dataset)", expanded=True):
        suggestions = [
            "How many records are there?",
            "Show distribution by " + (categorical_cols[0] if categorical_cols else df.columns[0]),
            "Show average " + numeric_cols[0] + " by " + (categorical_cols[0] if categorical_cols else df.columns[0]) if numeric_cols else "Show missing values by column",
            "Show top 10 " + (categorical_cols[0] if categorical_cols else df.columns[0]) + " by count",
            "Show monthly trend by " + date_cols[0] if date_cols else "Show unique values by " + (categorical_cols[0] if categorical_cols else df.columns[0]),
            "Show missing values",
            "Show correlation between " + numeric_cols[0] + " and " + numeric_cols[1] if len(numeric_cols) >= 2 else "Show maximum " + numeric_cols[0] if numeric_cols else "Show dataset summary",
        ]
        for i in range(0, len(suggestions), 2):
            c1,c2 = st.columns(2)
            c1.code(suggestions[i])
            if i+1 < len(suggestions):
                c2.code(suggestions[i+1])
    question = st.text_input("Your question", value="", placeholder="e.g. Show average sales by region, count records, or find missing values")
    if st.button("Analyze question", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.current_question = question
            try:
                if use_llm and api_key:
                    try:
                        plan = llm_answer(question, df, api_key, model)
                    except Exception as llm_exc:
                        st.warning(f"LLM unavailable; using local analysis instead ({type(llm_exc).__name__}).")
                        plan = infer_operation(question, df)
                else:
                    plan = infer_operation(question, df)
                plan = validate_operation(plan, df)
                filtered, result, summary = execute_plan(df, plan)
                st.session_state.last_result = result
                st.session_state.last_plan = plan
                st.session_state.last_question = question
                st.session_state.last_summary = summary
            except Exception as exc:
                st.error(f"Could not analyze question: {exc}")
                st.info("Try one of the suggested questions, or include a column name such as 'average revenue by region'.")
    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        st.markdown("---")
        st.subheader("Analysis results")
        st.write("**Question:**", st.session_state.last_question)
        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Result table")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("Download result CSV", result.to_csv(index=False).encode("utf-8"), "analytics_result.csv", "text/csv")
        with right:
            st.markdown("#### Chart visualization")
            fig = auto_chart(result, st.session_state.last_question)
            if fig is not None:
                fig.update_layout(template="plotly_white", height=400, margin=dict(l=10,r=10,t=60,b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rows to chart. Try a different question.")
        st.markdown("#### Explanation")
        st.info(st.session_state.last_summary)
        with st.expander("Validated analysis plan"):
            st.json(st.session_state.last_plan)

if page == "Dashboard":
    st.subheader("Automatic dataset dashboard")
    a,b,c,d = st.columns(4)
    a.metric("Records", f"{len(df):,}")
    b.metric("Numeric columns", len(numeric_cols))
    c.metric("Categorical columns", len(categorical_cols))
    d.metric("Date columns", len(date_cols))
    if categorical_cols:
        selected_cat = st.selectbox("Categorical column", categorical_cols, key="dash_cat")
        counts = df[selected_cat].astype(str).value_counts().head(25).rename_axis(selected_cat).reset_index(name="record_count")
        st.plotly_chart(px.bar(counts, x=selected_cat, y="record_count", text_auto=True, title=f"Record count by {selected_cat}"), use_container_width=True)
    if numeric_cols:
        selected_num = st.selectbox("Numeric column", numeric_cols, key="dash_num")
        vals = pd.to_numeric(df[selected_num], errors="coerce").dropna()
        if len(vals):
            hist = px.histogram(pd.DataFrame({selected_num: vals}), x=selected_num, nbins=25, title=f"Distribution of {selected_num}")
            st.plotly_chart(hist, use_container_width=True)
    if date_cols:
        selected_date = st.selectbox("Date column", date_cols, key="dash_date")
        dated = pd.to_datetime(df[selected_date], errors="coerce").dropna()
        if len(dated):
            trend = dated.dt.to_period("M").astype(str).value_counts().sort_index().rename_axis("month").reset_index(name="record_count")
            st.plotly_chart(px.line(trend, x="month", y="record_count", markers=True, title=f"Monthly records by {selected_date}"), use_container_width=True)

if page == "Dataset Explorer":
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    st.download_button("Download normalized dataset", df.to_csv(index=False).encode("utf-8"), "normalized_dataset.csv", "text/csv")
    schema = pd.DataFrame({
        "column": df.columns,
        "data_type": [str(df[c].dtype) for c in df.columns],
        "missing_values": [int(df[c].isna().sum()) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    st.markdown("#### Schema and data quality")
    st.dataframe(schema, use_container_width=True, hide_index=True)
    if numeric_cols:
        st.markdown("#### Numeric summary")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

if page == "SQL Explorer":
    st.subheader("🧮 SQL Explorer")
    st.write("Run read-only SQL queries against the currently loaded dataset using the table name `data`.")
    st.caption("Only SELECT queries are allowed.")
    sql_query = st.text_area("SQL query", value="SELECT * FROM data LIMIT 20;", height=120, key="sql_query")
    if st.button("Run SQL query", type="primary", use_container_width=True):
        try:
            import duckdb
            cleaned = sql_query.strip().rstrip(";")
            if not re.match(r"(?is)^select\\b", cleaned):
                raise ValueError("Only SELECT statements are allowed.")
            if re.search(r"(?is)\\b(insert|update|delete|drop|alter|create|attach|copy|install|load|pragma|export|import)\\b", cleaned):
                raise ValueError("This query contains a prohibited SQL operation.")
            conn = duckdb.connect(database=":memory:")
            conn.register("data", df)
            sql_result = conn.execute(cleaned).df()
            conn.close()
            st.session_state["sql_last_result"] = sql_result
            st.success(f"Query returned {len(sql_result):,} row(s).")
        except Exception as exc:
            st.error(f"SQL query failed: {exc}")
    if "sql_last_result" in st.session_state:
        sql_result = st.session_state["sql_last_result"]
        st.dataframe(sql_result, use_container_width=True, hide_index=True)
        if not sql_result.empty:
            st.download_button("Download SQL result CSV", sql_result.to_csv(index=False).encode("utf-8"), "sql_result.csv", "text/csv")
            sql_fig = auto_chart(sql_result, "SQL query result")
            if sql_fig is not None:
                sql_fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(sql_fig, use_container_width=True)

if page == "About":
    st.markdown("""
### How to use
1. Upload your own CSV or Excel file in the sidebar. A sample campus dataset loads automatically.
2. Review the generated questions. They adapt to the uploaded dataset's columns.
3. Ask a question and click **Analyze question**.
4. Review the result table, chart, explanation, and validated plan.

### Question types
- Counts: `How many records are there?`
- Grouped counts: `Show records by department`
- Numeric aggregates: `Average sales by region`, `sum of revenue by category`, `maximum score`
- Unique values: `How many unique customers?`
- Missing values: `Show missing values`
- Relationships: `Correlation between sales and profit`
- Filters: mention a category value from the dataset, e.g. `count orders from North`

### Important
The local parser supports common analytics patterns across different schemas; it cannot perfectly understand every possible phrasing without an LLM. For broader natural-language understanding, add a valid OpenAI API key in the sidebar. LLM output is converted to a constrained plan and is never executed as generated code. Charts are generated with Plotly for non-empty results.
""")
st.caption("Ask Your Data — AI Analytics Portal | Academic prototype")
