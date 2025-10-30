import json
import os
import subprocess
from collections import defaultdict

from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import PromptTemplate

import GIS_map

# Load API keys
os.environ["OPENAI_API_KEY"] = ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
serp = SerpAPIWrapper() if SERPAPI_API_KEY else None

# Configuration
PERSIST_HF = "../chroma_sip_csa_db[Huggingface Embedding]"
PERSIST_OPENAI = "../chroma_sip_csa_db[openai_embed3]"
COLLECTION_NAME = "sip_csa_chunks"
QUERY_LOG_PATH = "query_log.json"
TOP_K_DEFAULT = 5
MAX_CHAR_LIMIT = 80000

# Prompt template
# we can imporve the template
qa_prompt = PromptTemplate(
    input_variables=["context","question","external","user_context"],
    template="""
Context:
{context}

External Info:
{external}

User Context:
{user_context}

Question: {question}

Answer:
"""
)

# Utility functions -----------------------------------------------------------

# need to install ollama in local environment
def start_ollama():
    try:
        subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
            )
    except Exception as e:
        print(f"❌ Could not start Ollama: {e}")


def clean_text(text: str) -> str:
    return text.encode("utf-8","ignore").decode("utf-8")


def load_log():
    try:
        return json.load(open(QUERY_LOG_PATH))
    except:
        return {}


def save_log(log):
    with open(QUERY_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def top_queries(n=10):
    '''Return the n top queries from the query log'''
    log = load_log()
    freqs = defaultdict(int)
    for v in log.values():
        freqs[v.get("query","")] += 1
     # does this break if less than 10 queries? No but the .json file already stored more than 10
     # queries so shouldn't be a problem
    top = sorted(freqs.items(), key=lambda x: -x[1])[:n]
    return "\n".join(f"{i+1}. {q} — {c}x" for i,(q,c) in enumerate(top)) or "No queries."

# Global placeholders
retriever = None
summarizer = None
qa_chain = None
log = load_log()