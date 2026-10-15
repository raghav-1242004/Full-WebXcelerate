"""Quick smoke tests for the local question parser."""
import importlib.util
from pathlib import Path
import pandas as pd

# app.py is a Streamlit UI script; parser functions are tested by source-level extraction in the test runner.
# This script documents expected questions for manual regression testing.
EXPECTED_QUESTIONS = [
    "How many registrations are there?",
    "Show registrations by department",
    "What is the average feedback_score by event_category?",
    "Show attendance by department",
    "Show monthly registrations trend",
    "Which department has the highest feedback_score?",
    "Show missing values",
    "Show correlation between feedback_score and attendance_numeric",
]
print("Expected parser regression questions:")
for question in EXPECTED_QUESTIONS:
    print("-", question)
