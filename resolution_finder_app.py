# ----------------------------------------------------------
# Streamlit App: Resolution Finder (Exact Match + Top 5 Similar)
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from io import BytesIO
import base64
import openai
import os

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Incident Resolution Finder", layout="wide")

st.title("🚨 Incident Resolution Finder")
st.write("Search for resolutions from past incidents using summary and description.")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = ["summary", "description", "resolution notes"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=required).reset_index(drop=True)
    return df

# 👇 Change the filename if needed
data = load_data("Design Central Overall Cases Metrics (6).xlsx")

# -----------------------------
# Build TF-IDF retriever
# -----------------------------
@st.cache_resource
def build_retriever(df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df["resolution notes"].astype(str).tolist())
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(X)
    return vectorizer, nn

vectorizer, nn = build_retriever(data)

# -----------------------------
# Retrieval
# -----------------------------
def find_exact_matches(summary, description):
    return data[
        (data["summary"].str.lower() == summary.lower()) &
        (data["description"].str.lower() == description.lower())
    ]

def retrieve_similar(summary, description, top_k=5):
    query_text = (summary + " . " + description).strip()
    qv = vectorizer.transform([query_text])
    _, indices = nn.kneighbors(qv, n_neighbors=top_k)

    results = []
    for idx in indices[0]:
        row = data.iloc[idx]
        entry = {
            "Summary": row["summary"],
            "Description": row["description"],
            "Resolution": row["resolution notes"],
            "User": row.get("user", "N/A"),
            "User Story": row.get("user story", "N/A"),
            "Jira Status": row.get("jira status", "N/A"),
            "Jira Sprint": row.get("jira sprint", "N/A"),
            "Root Cause": row.get("root cause", "N/A"),
            "Resolution Category": row.get("resolution category", "N/A"),
            "Resolution Subcategory": row.get("resolution subcategory", "N/A"),
        }
        results.append(entry)
    return pd.DataFrame(results)

# -----------------------------
# Download link helper
# -----------------------------
def make_download_link(df, filename="results.csv", to_excel=False):
    if to_excel:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a download="{filename}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">📥 Download Excel</a>'
    else:
        csv = df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a download="{filename}" href="data:text/csv;base64,{b64}">📥 Download CSV</a>'
    return href

# -----------------------------
# LLM Insights (OpenAI)
# -----------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # set in Streamlit Cloud secrets or .env

def generate_llm_response(summary, description, sim_df):
    context = ""
    for _, row in sim_df.iterrows():
        context += f"""
Incident:
- Summary: {row['Summary']}
- Description: {row['Description']}
- Resolution: {row['Resolution']}
- Root Cause: {row['Root Cause']}
- Jira Status: {row['Jira Status']}
- Jira Sprint: {row['Jira Sprint']}
- Resolution Category: {row['Resolution Category']}
- Resolution Subcategory: {row['Resolution Subcategory']}
---
"""

    prompt = f"""
A user reported an incident with:
- Summary: {summary}
- Description: {description}

Based on past similar incidents:
{context}

Provide:
1. A likely resolution summary
2. The most probable root cause
3. A clear insight for stakeholders
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"

# -----------------------------
# UI
# -----------------------------
summary = st.text_input("Summary", placeholder="Enter issue summary")
description = st.text_area("Description", placeholder="Enter issue description")

if st.button("🔍 Find Resolution"):
    if not summary or not description:
        st.warning("Please enter both summary and description.")
    else:
        # Exact matches
        exact_df = find_exact_matches(summary, description)
        if not exact_df.empty:
            st.subheader("🟢 Exact Same Incident(s) Found Previously")
            st.dataframe(exact_df)

        # Similar matches
        sim_df = retrieve_similar(summary, description, top_k=5)
        st.subheader("🔍 Top 5 Similar Incidents")
        st.dataframe(sim_df)

        # Download buttons
        st.markdown(make_download_link(sim_df, "similar_results.csv", to_excel=False), unsafe_allow_html=True)
        st.markdown(make_download_link(sim_df, "similar_results.xlsx", to_excel=True), unsafe_allow_html=True)

        # LLM Insights
        st.subheader("🤖 LLM Insights")
        llm_response = generate_llm_response(summary, description, sim_df)
        st.markdown(llm_response)
