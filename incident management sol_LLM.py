import os, base64, requests, openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import ipywidgets as widgets
from IPython.display import display, Markdown
from io import BytesIO
import base64 as b64

from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
client_id = "cG9jLXRyaWFsMjAyM1NlcHRlbWJlcjE5_47e58d62e9b3b152bf9eb5d8c079e5"
client_secret = "5pwo_yWk5Jy4gESDzJf9OTnbD8LMnA1wGaEjSjA65gvsrLFxr8FwHLrZ1aVbO2vE"
APP_KEY = "egai-prd-sco-prdops-1"

def get_access_token(client_id, client_secret):
    access_token_url = "https://id.cisco.com/oauth2/default/v1/token"
    payload = "grant_type=client_credentials"
    value = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {value}",
    }
    token_response = requests.post(access_token_url, headers=headers, data=payload)
    return token_response.json()["access_token"]
# Configure OpenAI (Cisco)
openai.api_key = get_access_token(client_id, client_secret)
openai.api_base = "https://chat-ai.cisco.com"
openai.api_type = "azure"
openai.api_version = "2023-08-01-preview"

llm = ChatOpenAI(
    openai_api_base=openai.api_base,
    openai_api_key=openai.api_key,
    temperature=0,
    model_kwargs={
        "engine": "gpt-4o",
        "stop": ["<|im_end|>"],
        "user": f'{{"appkey": "{APP_KEY}"}}',
    },
)

# Step 1. Load Excel
# -----------------------------
PATH = "Design Central Overall Cases Metrics (6).xlsx"
data = pd.read_excel(PATH)
data.columns = [c.strip().lower() for c in data.columns]

summary_col = "summary"
desc_col = "description"
resolution_col = "resolution notes"

meta_cols = [
    "user", "user story", "jira status", "jira sprint",
    "root cause", "resolution category", "resolution subcategory"
]

data = data.dropna(subset=[summary_col, desc_col, resolution_col]).reset_index(drop=True)


# Step 2. Build TF-IDF retriever
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(data[resolution_col].astype(str).tolist())
nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(X)

# -----------------------------
# Retrieval functions
# -----------------------------
def find_exact_matches(summary, description):
    return data[
        (data[summary_col].str.lower() == str(summary).lower()) &
        (data[desc_col].str.lower() == str(description).lower())
    ]

def retrieve_similar(summary, description, top_k=5):
    query_text = (str(summary) + " . " + str(description)).strip()
    qv = vectorizer.transform([query_text])
    distances, indices = nn.kneighbors(qv, n_neighbors=top_k)
    sims = 1 - distances[0]

    results = []
    for idx, sim in zip(indices[0], sims):
        row = data.iloc[idx]
        entry = {
            "similarity": float(sim),
            "source_summary": str(row[summary_col]),
            "source_description": str(row[desc_col]),
            "resolution": str(row[resolution_col]),
        }
        for col in meta_cols:
            entry[col] = str(row[col]) if col in data.columns else "N/A"
        results.append(entry)
    return pd.DataFrame(results)

def make_download_link(df, filename="results.csv", to_excel=False):
    if to_excel:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        b64_data = b64.b64encode(buffer.getvalue()).decode()
        href = f'<a download="{filename}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_data}">📥 Download Excel</a>'
    else:
        csv = df.to_csv(index=False).encode()
        b64_data = b64.b64encode(csv).decode()
        href = f'<a download="{filename}" href="data:text/csv;base64,{b64_data}">📥 Download CSV</a>'
    return href



# LLM Insights
# -----------------------------
def generate_llm_response(summary, description, sim_df):
    context = ""
    for _, row in sim_df.iterrows():
        context += (
            f"Incident:\n"
            f"- Summary: {row['source_summary']}\n"
            f"- Description: {row['source_description']}\n"
            f"- Resolution: {row['resolution']}\n"
            f"- Root Cause: {row.get('root cause','N/A')}\n"
            f"- Jira Status: {row.get('jira status','N/A')}\n"
            f"- Jira Sprint: {row.get('jira sprint','N/A')}\n"
            f"- Resolution Category: {row.get('resolution category','N/A')}\n"
            f"- Resolution Subcategory: {row.get('resolution subcategory','N/A')}\n"
            "---\n"
        )
    prompt = (
        f"A user reported an incident with:\n"
        f"- Summary: {summary}\n"
        f"- Description: {description}\n\n"
        f"Based on the past similar incidents below:\n{context}\n"
        "Provide:\n"
        "1. A likely resolution summary.\n"
        "2. The most probable root cause.\n"
        "3. An insight for stakeholders (in plain English).\n"
    )

    try:
        response = llm([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"



# Widgets UI
# -----------------------------
summary_box = widgets.Text(
    value='',
    placeholder='Enter issue summary',
    description='Summary:',
    layout=widgets.Layout(width="70%")
)

desc_box = widgets.Textarea(
    value='',
    placeholder='Enter issue description',
    description='Description:',
    layout=widgets.Layout(width="70%", height="100px")
)

button = widgets.Button(description="Find Resolution", button_style='success')
output = widgets.Output()

def on_click(b):
    with output:
        output.clear_output()
        summary = summary_box.value.strip()
        description = desc_box.value.strip()
        


        if not summary and not description:
            display(Markdown("⚠️ Please enter a summary or description."))
            return

        exact_df = find_exact_matches(summary, description)
        if not exact_df.empty:
            display(Markdown("## 🟢 Exact Same Incident(s) Found Previously"))
            for _, row in exact_df.iterrows():
                display(Markdown(f"- **Resolution:** {row[resolution_col]}"))
                for col in meta_cols:
                    if col in exact_df.columns:
                        display(Markdown(f"  - **{col.title()}:** {row[col]}"))
                display(Markdown("---"))

        sim_df = retrieve_similar(summary, description, top_k=5)
        if not sim_df.empty:
            display(Markdown("## 🔍 Top 5 Similar Incidents"))
            display(sim_df)

            display(Markdown(make_download_link(sim_df, "similar_results.csv", to_excel=False)))
            display(Markdown(make_download_link(sim_df, "similar_results.xlsx", to_excel=True)))

            display(Markdown("## 🤖 LLM Insights"))
            llm_response = generate_llm_response(summary, description, sim_df)
            display(Markdown(llm_response))
        else:
            display(Markdown("⚠️ No similar resolutions found."))

button.on_click(on_click)
ui = widgets.VBox([summary_box, desc_box, button, output])
display(ui)

