import os
import json
from datetime import datetime
from collections import defaultdict
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents.map_reduce import create_map_reduce_chain
from langchain_classic.chains.summarize import load_summarize_chain

from langchain_classic.chains.llm import LLMChain
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.documents import Document as LCDocument

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from chromadb.config import Settings
import subprocess


# Load API keys
os.environ["OPENAI_API_KEY"] = ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
serp = SerpAPIWrapper() if SERPAPI_API_KEY else None

# Configuration
PERSIST_HF = "/content/drive/MyDrive/CalWorks/Vector Database/Output/chroma_sip_csa_db[Huggingface Embedding]"
PERSIST_OPENAI = "/content/drive/MyDrive/CalWorks/Vector Database/Output/chroma_sip_csa_db[openai_embed3]"
COLLECTION_NAME = "sip_csa_chunks"
QUERY_LOG_PATH = "/content/drive/MyDrive/CalWorks/Vector Database/Output/query_log.json"
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
    json.dump(log, open(QUERY_LOG_PATH, "w"), indent=2)


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

# Initialization function supporting Ollama and OpenAI
def init_engine(embed_backend, embed_model, llm_backend, llm_model):
    '''Initialize engine suppporting Ollama and OpenAI

    Inputs:
        embed_backend (str): which backend to use (MiniLM or OpenAI Embeddings)
        embed_model (str): embed model name
        llm_backend (str): which llm backend to use (OpenAI or Ollama)
        llm_model (str): llm model name

    Returns:
        (None) updates the global variables retriever, summarizer and qa_chain
    '''
    # Update the global version of the variable
    global retriever, summarizer, qa_chain

    # Use MiniLM if avaiable, otherwise use open AI
    if embed_backend == "MiniLM":
        embedder = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"})
    else:
        embedder = OpenAIEmbeddings(
            model=embed_model,
            openai_api_key=OPENAI_API_KEY)

    # Set up retriever to vector database
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=(PERSIST_HF if embed_backend == "MiniLM" else PERSIST_OPENAI),
        embedding_function=embedder
)
    retriever = store.as_retriever()

    if llm_backend == "OpenAI":
        llm = ChatOpenAI(
            model_name=llm_model,
            temperature=0,
            openai_api_key=OPENAI_API_KEY
            )
    else:
      # temperature 0 so no hallucination
        llm = Ollama(model=llm_model, temperature=0)

    summarizer = load_summarize_chain(llm, chain_type="map_reduce")
    # splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # summarizer = create_map_reduce_chain(llm, text_splitter=splitter)
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)


# Document summarization

def summarize_docs(docs):
    """For list of documents, summarize them into one single summary.

    Inputs:
        docs (list): List of documents to summarize

    Returns (str) summary
    """
    pages = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        header = f'''[{i+1}] {
            meta.get('county','Unknown')
            } | Section: {
            meta.get('section','?')
            }'''
        pages.append(LCDocument(page_content=header+"\n"+doc.page_content))
    return summarizer.run(pages)


# File analysis (text only)

def extract_text_from_file(path):
    return open(path, 'r').read()


def analyze_file(file_path, query=""):
    '''Analyze the file (currently only returns the text in the file?)'''
    # i think this function only return the first 80000 characters now
    # definitely need to be improve
    text = extract_text_from_file(file_path)
    return text[:MAX_CHAR_LIMIT]


# Main QA function
def ask(
        query,
        k,
        use_ext,
        ext_query,
        embed_backend,
        embed_model,
        llm_backend,
        llm_model
        ):
    '''Query the LLM using the embed/llm models provided

    Inputs:
        query (str): The question asked of the LLM
        k (int): Number of most probable tokens (for top-k sampling)
        use_ext (bool): Use external query or not
        ext_query (str): External query
        embed_backend (str): embedding backend to use (MiniLM or OpenAI
            Embeddings)
        embed_model (str): embed model name
        llm_backend (str): llm backed to use (Ollama or OpenAI)
        llm_model (str): llm model name

    Returns:
        (tuple) of:
            response (str),
            top queries (list of strings),
            external response (str)
        '''
    global log
    # if retriever is None:
    init_engine(embed_backend, embed_model, llm_backend, llm_model)

    docs = retriever.get_relevant_documents(query, k=k)
    if not docs:
      # better communication to users?
        return "No docs found.", "", ""

    summary = summarize_docs(docs)

    external = ""
    if use_ext:
        if not serp:
            external = "[Web search disabled]"
        elif ext_query:
            try:
                external = serp.run(ext_query)
            except Exception as e:
                external = f"[Web search error: {e}]"

    resp = qa_chain.invoke({
        "context": clean_text(summary),
        "question": clean_text(query),
        "external": clean_text(external),
        "user_context": f'''Counties: {', '.join(
            {d.metadata.get('county','') for d in docs})
            }'''
    })["text"]

    t = datetime.now().isoformat()
    log[t] = {"query": query}
    save_log(log)

    excerpts = "\n\n---\n\n".join([
        f'''[{i+1}] 📍 {
            doc.metadata.get('county', 'Unknown')
            } | {
            doc.metadata.get('report_type', 'Unknown')
            } | Section: {
            doc.metadata.get('section', 'Unknown')
            } | Page {
            doc.metadata.get('page', '?')
            }\n{doc.page_content.strip()}'''
        for i, doc in enumerate(docs)
    ])

    full_response = resp.strip() + "\n\n📚 Used Excerpts:\n\n" + excerpts
    return full_response, top_queries(), external.strip()
