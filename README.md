# Ask Your Data — AI Analytics Portal (Fixed)

A clean, schema-aware natural-language analytics app for the Problem 6 requirements.

## Features
- CSV, XLSX, XLS upload.
- Automatic column normalization and lightweight type detection.
- Suggested questions generated from the uploaded dataset's column types.
- Safe local analytics: counts, grouped counts, mean, median, sum, min/max, distinct values, missing values, correlations.
- Result table and Plotly chart for non-empty results.
- Sidebar navigation: Ask Your Data, Dashboard, Dataset Explorer, SQL Explorer, and About.
- Read-only SQL Explorer using DuckDB; only SELECT queries are allowed.
- Auto dashboard charts for categorical, numeric, and date columns.
- Optional OpenAI API integration for broader natural-language interpretation. No API key is required for common supported questions.
- CSV downloads for results and normalized dataset.
- Synthetic campus event dataset (150 rows) included.

## Windows setup
Install Python 3.10+ from https://www.python.org/downloads/windows/ and select **Add Python to PATH** during installation.

Open this folder in VS Code. Run in the terminal:

```bat
py --version
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

If `py` is unavailable but `python --version` works, replace `py` with `python`.

## Optional LLM
Enter an OpenAI API key in the app sidebar and enable "Use LLM for question interpretation". API use may incur charges. Keep keys private. If LLM fails, the app falls back to the local parser.

## Supported dataset types
CSV / Excel with mixed categorical, numeric, and date columns. The app normalizes column names, detects numeric-like fields and dates, and builds suggestions based on the loaded schema.

## Limitations
Natural-language questions can be phrased in many ways. Local mode supports common question patterns; for more flexible interpretation use the optional LLM. Always check the displayed plan and result before relying on an answer.
