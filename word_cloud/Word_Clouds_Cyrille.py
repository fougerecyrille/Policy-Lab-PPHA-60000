########################################
# CalWORKs CSA Policy Analysis Pipeline
# --------------------------------------
# Features:
# - Robust PDF extraction (pdfplumber → PyPDF2 fallback)
# - Preserves original stopwords & KEEP_TERMS
# - Defensive word statistics (no KeyError on empty input)
# - Avoids shadowing of pathlib.Path
# - Clear debug logging
# - WordCloud visualizations for TOP-20 terms
# - Sentence-level and word-level sentiment analysis
# - NEW: Computes sentence-level sentiment per Top-20 word per county
########################################

import os
import re
import zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import unicodedata
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Quiet noisy PDF warnings (pdfminer/ghostscript style)
# -------------------------------
import logging
for noisy in (
    "pdfminer",
    "pdfminer.pdfinterp",
    "pdfminer.cmapdb",
    "pdfminer.psparser",
    "pdfminer.pdfpage"
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# -------------------------------
# Optional: DOCX readers (kept for completeness)
# -------------------------------
import docx2txt
from docx import Document

# -------------------------------
# Optional: gensim (Word2Vec) support
# -------------------------------
W2V_OK = True
try:
    from gensim.models import Word2Vec, KeyedVectors
    from gensim.downloader import load as gensim_load
except Exception:
    W2V_OK = False

# -------------------------------
# Optional: PDF readers
# -------------------------------
_PDF_BACKENDS = {}
try:
    import pdfplumber
    _PDF_BACKENDS["pdfplumber"] = True
except Exception:
    _PDF_BACKENDS["pdfplumber"] = False

try:
    import PyPDF2
    _PDF_BACKENDS["pypdf2"] = True
except Exception:
    _PDF_BACKENDS["pypdf2"] = False

# -------------------------------
# 0. Ensure NLTK resources are available
# -------------------------------
def _safe_download(resource_path: str, download_name: str):
    """
    Ensure an NLTK resource exists; download if missing.
    
    Parameters
    ----------
    resource_path : str
        Path used internally by NLTK to check for the resource.
    download_name : str
        Name of the NLTK package to download if missing.
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name)

# Download essential NLTK packages
_safe_download("tokenizers/punkt", "punkt")
_safe_download("tokenizers/punkt_tab", "punkt_tab")
_safe_download("corpora/stopwords", "stopwords")
_safe_download("sentiment/vader_lexicon", "vader_lexicon")

# -------------------------------
# 1.1 USER CONFIGURATION
# -------------------------------

# Path to ZIP file containing CalWORKs CSAs
ZIP_PATH =  # Edit the file path so that it points to the .zip file where all the different reports from the various California counties are stored; the word clouds and 
# sentiment analysis will be performed on the reports stored in the aforementioned .zip file
BASE_DIR = Path(ZIP_PATH).parent

# Folder where ZIP will be extracted
EXTRACT_DIR = BASE_DIR / "CalWORKs data_extracted"
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Output folder for WordClouds and results
OUTPUT_DIR = BASE_DIR / "Word Clouds Update"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1.2 Sanity check of paths
# -------------------------------
print("=== Path Verification ===")
print("ZIP exists:", Path(ZIP_PATH).exists())
print("Extract dir:", EXTRACT_DIR)
print("Output dir:", OUTPUT_DIR)
print("=========================")

# -------------------------------
# 1.3 Extract ZIP once (if not already extracted)
# -------------------------------
if (not any(EXTRACT_DIR.rglob("*"))) and Path(ZIP_PATH).exists():
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
print(f"Extracted ZIP to: {EXTRACT_DIR}")

# -------------------------------
# 1.4 List PDF files to analyze
# -------------------------------
pdf_files = list(EXTRACT_DIR.rglob("*.pdf"))
print("Files to analyze:")
for f in pdf_files:
    print(" -", f.relative_to(EXTRACT_DIR))

# -------------------------------
# 1.5 Confirm output directory exists
# -------------------------------
print(f"\nOutput folder ready: {OUTPUT_DIR}\n")

# ----------------------------------------------------
# 2. TEXT READER (PDF / DOCX)
# ----------------------------------------------------
# Features:
# - PDF reading fallback chain: pdfplumber → PyMuPDF (fitz) → PyPDF2 → OCR (first few pages)
# - DOCX reading fallback: python-docx → docx2txt
# - Automatic dispatch by file extension
# ----------------------------------------------------

# 2.1 Check for optional PyMuPDF (fitz) and OCR dependencies
_HAS_FITZ = False
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    pass

_HAS_OCR = False
try:
    import pytesseract
    from PIL import Image
    _HAS_OCR = True
except Exception:
    pass


# 2.2 Lightweight OCR for first `max_pages` of PDF
def _ocr_pdf_pages_to_text(path: Path, max_pages: int = 5) -> str:
    """
    Rasterize the first `max_pages` of a PDF and extract text via OCR.
    
    Parameters
    ----------
    path : Path
        Path to PDF file.
    max_pages : int
        Maximum number of pages to OCR.
    
    Returns
    -------
    str
        Extracted text, or empty string if OCR unavailable.
    """
    if not (_HAS_OCR and _HAS_FITZ):
        return ""

    text_parts = []
    try:
        doc = fitz.open(str(path))
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            try:
                t = pytesseract.image_to_string(img)
            except Exception:
                t = ""
            if t.strip():
                text_parts.append(t)
    except Exception:
        return ""

    return "\n".join(text_parts)


# 2.3 PDF Reader with multiple fallback methods
def read_pdf_text(path: Path) -> str:
    """
    Read text from a PDF using the following fallback chain:
    pdfplumber → PyMuPDF (fitz) → PyPDF2 → OCR (first few pages).
    
    Parameters
    ----------
    path : Path
        Path to PDF file.
    
    Returns
    -------
    str
        Extracted text or empty string if no text is found.
    """
    # --- 1) pdfplumber ---
    if _PDF_BACKENDS.get("pdfplumber", False):
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                parts = [page.extract_text() or "" for page in pdf.pages if (page.extract_text() or "").strip()]
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → pdfplumber OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → pdfplumber empty.")
        except Exception as e:
            print(f"[PDF] pdfplumber failed on {path.name}: {e}")

    # --- 2) PyMuPDF (fitz) ---
    if _HAS_FITZ:
        try:
            doc = fitz.open(str(path))
            parts = [page.get_text("text") for page in doc if (page.get_text("text") or "").strip()]
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → PyMuPDF OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → PyMuPDF empty.")
        except Exception as e:
            print(f"[PDF] PyMuPDF failed on {path.name}: {e}")

    # --- 3) PyPDF2 ---
    if _PDF_BACKENDS.get("pypdf2", False):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(path))
            parts = [page.extract_text() or "" for page in reader.pages if (page.extract_text() or "").strip()]
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → PyPDF2 OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → PyPDF2 empty.")
        except Exception as e:
            print(f"[PDF] PyPDF2 failed on {path.name}: {e}")

    # --- 4) OCR fallback ---
    ocr_text = _ocr_pdf_pages_to_text(path, max_pages=5)
    if ocr_text.strip():
        print(f"[PDF] {path.name} → OCR sample OK ({len(ocr_text)} chars; first 5 pages)")
        return ocr_text

    # --- No text found ---
    print(f"[PDF] {path.name} → NO TEXT (scanned/secured?)")
    return ""


# 2.4 DOCX Reader with python-docx → docx2txt fallback
def read_docx_text(path: Path) -> str:
    """
    Extract text from a DOCX file, falling back from python-docx to docx2txt.
    
    Parameters
    ----------
    path : Path
        Path to DOCX file.
    
    Returns
    -------
    str
        Extracted text (empty string if none found).
    """
    text1 = ""
    try:
        from docx import Document
        doc = Document(str(path))
        text1 = "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        text1 = ""

    if len(text1.strip()) < 50:
        try:
            import docx2txt
            text2 = docx2txt.process(str(path))
            if len(text2.strip()) > len(text1.strip()):
                return text2
        except Exception:
            pass

    return text1


# 2.5 Dispatch function to read any text file
def read_any_text(path: Path) -> str:
    """
    Dispatch text reading by file extension.
    
    Parameters
    ----------
    path : Path
        Path to the file.
    
    Returns
    -------
    str
        Extracted text for PDF or DOCX, empty string for other types.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf_text(path)
    elif ext == ".docx":
        return read_docx_text(path)
    else:
        return ""

# ----------------------------------------------------
# 3. STOPWORDS (policy-tuned) + GEO TERMS + LEMMATIZATION
# ----------------------------------------------------
# Features:
# - Custom stopwords combining NLTK base, policy/admin terms, organizational terms, geo terms, and filler words
# - Keeps important policy-relevant terms intact (KEEP_TERMS)
# - Word lemmatization for noun → verb → adjective
# - Token cleaning and filtering
# ----------------------------------------------------

from nltk.stem import WordNetLemmatizer

def build_stopwords() -> set:
    """
    Build a comprehensive, policy-tuned stopword set for CSA document analysis.

    Combines:
        - Standard NLTK English stopwords
        - Administrative / metrics terms
        - Organizational terms
        - Geographic terms
        - Common report filler words
    Excludes KEEP_TERMS to preserve policy-relevant vocabulary.
    """
    base_sw = set(stopwords.words('english'))

    # 3.1 Administrative / metrics terms
    admin_metrics = {
        "achieve", "analysis", "app", "applicable", "appraisal", "application", "applications", "apps",
        "assessment", "assistance", "assist", "attendance", "average", "business", "call",
        "case", "cases", "communication", "comparison", "compared", "complete", "completion",
        "contact", "cycle", "data", "datas", "decrease", "documentation", "eligibility", "eligible",
        "employmentrate", "engagementrate", "exit", "feedback", "improve", "improvement",
        "improvements", "improving", "increase", "management", "measure", "measures",
        "median", "meet", "meeting", "meetings", "opportunities", "opportunity",
        "orientation", "outcomes", "participationrate", "percent", "percentage", "performance",
        "period", "phone", "plan", "plans", "practice", "practices", "process", "procedure",
        "procedures", "program", "programs", "referral", "referrals", "requirement",
        "requirements", "resource", "resources", "results", "review", "reviewed", "reviews",
        "service", "services", "sip", "sps", "strategy", "strategies", "support", "supportive",
        "supports", "timeline", "timeliness", "wtw", "population"
    }

    # 3.2 Organizational / program terms
    org_terms = {
        "agencies", "agency", "caseworker", "caseworkers", "cbos", "cdss", "cct", "center", "centers",
        "client", "clients", "collaborator", "collaborators", "customer", "customers",
        "calsaw", "calsaws", "calworks", "cwd", "debs", "department", "departments",
        "des", "dess", "dha", "doe", "dss", "ecm", "ecms", "edc", "ess", "esss", "etw", "hhs",
        "hsa", "icdss", "kchsa", "manager", "managers", "mcdss", "oar", "ocat", "office", "offices",
        "ore", "ota", "partner", "partners", "participant", "participants", "programmatic",
        "provider", "providers", "recipient", "recipients", "sfhsa", "specialist", "specialists",
        "staff", "staffed", "staffing", "system", "tad", "team", "teams", "ts", "unit", "units",
        "vendor", "vendors", "vchsa", "ychhsd", "welfare", "wex"
    }   

    # 3.3 Geographic terms (generic + county tokens)
    geo_terms = {
        "area", "areas", "bay", "cal", "california", "central", "city", "cities",
        "coast", "coastal", "county", "counties", "district", "districts", "inland",
        "metro", "northern", "region", "regional", "southern", "valley", "valleys",
        "barbara", "benito", "bernardino", "clara", "contra", "costa", "cruz", "de", "del",
        "dorado", "diego", "el", "francisco", "glenn", "humboldt", "imperial", "inyo",
        "joaquin", "kern", "kings", "lake", "lassen", "la", "luis", "marin", "mariposa",
        "mateo", "mendocino", "merced", "mono", "monterey", "napa", "nevada", "obispo",
        "orange", "placer", "plumas", "riverside", "sacramento", "san", "santa", "shasta",
        "sierra", "siskiyou", "solano", "sonoma", "stanislaus", "sutter", "tehama", "trinity",
        "tulare", "tuolumne", "ventura", "yolo", "yuba", "alameda", "alpine", "amador",
        "angeles", "butte", "calaveras", "colusa", "fresno", "madera", "modoc", "norte", "los"
    }

    # 3.4 Common report filler words
    report_filler = {
        "additional", "additionally", "also", "among", "amongst", "and", "available",
        "bad", "best", "cause", "caused", "causes", "consider", "considered", "considering",
        "continue", "continued", "continuing", "continuum", "could", "current",
        "daily", "day", "determine", "determined", "determines", "different",
        "due", "eight", "example", "examples", "exited", 
        "figure", "five", "focused", "focusing", "focus", "four",
        "further", "good", "group", "higher", "however", "identify", "identified",
        "identifying", "impact", "impacted", "impacting", "impacts", "information",
        "initial", "internal", "issue", "january", "february", "march", "april", "may",
        "june", "july", "august", "september", "october", "november", "december",
        "lower", "month", "monthly", "months", "might", "new", "no", "note", "noted",
        "number", "numbers", "observed", "often", "ongoing", "overall", "particularly",
        "particular", "part", "parts", "people", "person", "persons", "post",
        "receive", "regard", "regarding", "regards", "relating", "related", "relatedly",
        "report", "respectively", "result", "resulted", "self", "since", "six", "seven",
        "so", "specific", "specifically", "step", "tab", "table", "ten", "three",
        "time", "typically", "typical", "two", "use", "variety", "various", "within",
        "well", "yes", "year", "yearly", "years", "one", "toward", "address",
        "section", "social", "high", "individual", "need", "etc", "rate"
    }

    # 3.5 Keep important policy-relevant terms
    KEEP_TERMS = {
        "employment","employ","employed","job","jobs","work","workforce","wage","wages","income","earnings",
        "engagement","engage","engaged","participation","participate","sanction","sanctions","sanctioned",
        "barrier","barriers","challenge","challenges","housing","homeless","homelessness","rent","cost","costs","poverty",
        "transportation","childcare","child","children","care","caregiver","family","families",
        "domestic","abuse","violence","safety","mental","health","healthcare","behavioral","behavioralhealth",
        "language","bilingual","spanish","english","access","accessibility","disparities","equity","racial","race","racism",
        "covid","pandemic","public","benefits","benefit","stabilization","stabilize","stabilizing",
        "technology","digital","internet","immigrant","immigrants","refugee","refugees","rural","urban",
        "parent","parents","parenting","eviction","evictions","bus","transit","equitable"
    }

    # Combine all sets and exclude KEEP_TERMS
    sw = base_sw.union(admin_metrics, org_terms, geo_terms, report_filler)
    sw = {w for w in sw if w not in KEEP_TERMS}
    return sw

# Instantiate STOPWORDS set
STOPWORDS = build_stopwords()

# ----------------------------------------------------
# 3.6 Lemmatization setup
# ----------------------------------------------------
_safe_download("corpora/wordnet", "wordnet")
_safe_download("corpora/omw-1.4", "omw-1.4")

LEM = WordNetLemmatizer()

def lemmatize_token(token: str) -> str:
    """
    Lemmatize a token in the sequence: noun → verb → adjective.
    Returns lemmatized token as string.
    """
    t = LEM.lemmatize(token, pos='n')
    t = LEM.lemmatize(t, pos='v')
    t = LEM.lemmatize(t, pos='a')
    return t

# ----------------------------------------------------
# 3.7 Clean and tokenize text
# ----------------------------------------------------
def clean_and_tokenize(text: str) -> list:
    """
    Clean text, tokenize, lemmatize, and filter stopwords.
    
    Steps:
    1. Lowercase the text
    2. Replace hyphens and slashes with spaces
    3. Remove non-alphabetic characters
    4. Collapse multiple spaces
    5. Tokenize
    6. Lemmatize each token
    7. Remove stopwords, short tokens (<3 chars), and numeric tokens
    """
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatize_token(t) for t in tokens]
    cleaned = [t for t in lemmas if t not in STOPWORDS and len(t) > 2 and not t.isnumeric()]
    return cleaned

# ----------------------------------------------------
# 4. SYNONYM GROUPING (Thesaurus Collapse)
# ----------------------------------------------------
def build_thesaurus():
    """
    Build a mapping of semantically similar words (variants) to a canonical token.
    This helps consolidate related terms (e.g., "jobs", "employed", "workforce")
    into a single root concept (e.g., "employment") for consistent counting and plotting.

    Returns:
        tuple:
            groups (dict): canonical term → set of related variants
            v2c (dict): variant → canonical term (inverse mapping)
    """
    groups = {
        # Employment and labor-related concepts
        "employment": {"employment", "employ", "employed", "job", "jobs", "work", "workforce"},

        # Income and pay concepts
        "wages": {"wage", "wages", "pay", "earnings", "income"},

        # Engagement and participation
        "engagement": {"engagement", "engage", "engaged", "participation", "participate",
                       "outreach", "attendance", "orientation"},

        # Sanctions, compliance, and penalties
        "sanctions": {"sanction", "sanctions", "sanctioned", "compliance",
                      "penalty", "penalties", "noncompliance", "good", "cause", "goodcause"},

        # Housing and homelessness
        "housing": {"housing", "rent", "rents", "eviction", "evictions", "homeless",
                    "homelessness", "shelter"},

        # Childcare and family support
        "childcare": {"childcare", "child", "children", "care", "parent", "parents",
                      "parenting", "caregiver"},

        # Transportation and commuting
        "transportation": {"transportation", "bus", "buses", "transit", "distance", "commute"},

        # Mental and behavioral health
        "mental_health": {"mental", "health", "healthcare", "behavioral",
                          "counseling", "therapy"},

        # Language access and translation
        "language_access": {"language", "bilingual", "spanish", "english",
                            "interpreter", "translation", "access", "accessibility"},

        # Equity and racial disparities
        "equity": {"equity", "equitable", "disparities", "racial", "race", "racism"},

        # Poverty and cost of living
        "poverty": {"poverty", "lowincome", "low", "income", "cost", "costs"},

        # Domestic violence and safety
        "violence": {"domestic", "violence", "abuse", "dv", "safety"},

        # Technology and digital access
        "technology": {"technology", "digital", "internet", "online", "device", "devices"},

        # Immigration and refugee concepts
        "immigration": {"immigrant", "immigrants", "refugee", "refugees"},
    }

    # Inverse mapping: variant → canonical term
    v2c = {variant: canon for canon, variants in groups.items() for variant in variants}

    return groups, v2c


# Build thesaurus dictionaries
THESAURUS, VAR2CANON = build_thesaurus()


def collapse_tokens_to_canon(tokens: list[str]) -> list[str]:
    """
    Replace each token by its canonical representative (if it exists in the thesaurus).
    If a token is not found, it remains unchanged.

    Args:
        tokens (list[str]): list of lemmatized tokens

    Returns:
        list[str]: tokens collapsed to canonical forms
    """
    return [VAR2CANON.get(t, t) for t in tokens]

# ----------------------------------------------------
# 5. WORD STATISTICS & VISUALIZATION (WordCloud + Top-20)
# ----------------------------------------------------
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import matplotlib.colors as mcolors


def compute_word_stats(tokens: list[str]) -> pd.DataFrame:
    """
    Compute frequency statistics for a list of tokens.
    """
    total = len(tokens)
    if total == 0:
        return pd.DataFrame(columns=["word", "freq", "relative_freq"])

    freq = Counter(tokens)
    rows = [{"word": w, "freq": c, "relative_freq": c / total} for w, c in freq.items()]
    df = pd.DataFrame(rows).sort_values("freq", ascending=False).reset_index(drop=True)
    return df

import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors
import numpy as np

def frequency_color_func(freq_dict):
    """
    Retourne une fonction de couleur basée sur la fréquence relative
    d'un mot : plus le mot est fréquent, plus la couleur est foncée.
    """
    # Normaliser les fréquences pour qu'elles soient entre 0 et 1
    max_freq = max(freq_dict.values())
    min_freq = min(freq_dict.values())
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        freq = freq_dict[word]
        # Plus le mot est fréquent → valeur plus élevée → couleur plus foncée
        norm = (freq - min_freq) / (max_freq - min_freq + 1e-5)  # éviter division par 0
        # Gradient bleu clair → bleu foncé
        return mcolors.to_hex((0.658*(1-norm)+0.0*norm, 0.792*(1-norm)+0.2*norm, 1*(1-norm)+0.2*norm))
    return color_func

def infer_county_name(path: str) -> str:
    """
    Extract a concise, human-readable county name from a filename.
    """
    base = os.path.basename(path)
    base_noext = os.path.splitext(base)[0]
    label = base_noext.replace("_", " ").strip()
    words = label.split()
    if len(words) > 7:
        label = " ".join(words[:7])
    return label

def build_wordcloud(freq_df, outpath, title=None, top_n=20, show=False):
    """
    Generate and save (or display) a WordCloud where word size and color
    reflect their relative frequency in the text.

    The title explicitly states that this represents the top-N most frequent terms.

    Args:
        freq_df (pd.DataFrame): DataFrame with columns 'word' and 'freq'.
        outpath (str): Path where the generated image will be saved.
        title (str, optional): Report title (usually inferred from filename).
        top_n (int, optional): Number of top words to include (default = 20).
        show (bool, optional): If True, display instead of saving the figure.
    """
    if freq_df.empty:
        print(f"[WARN] No words to plot for {outpath}")
        return

    # --- Keep only top-N words ---
    freq_df_top = freq_df.head(top_n)
    freq_dict = dict(zip(freq_df_top["word"], freq_df_top["freq"]))

    # --- Custom color gradient: light blue → dark blue ---
    blues = LinearSegmentedColormap.from_list("custom_blues", ["#a8caff", "#003366"])

    # --- Generate word cloud ---
    wc = WordCloud(
        width=1200,
        height=900,
        background_color="white",
        prefer_horizontal=0.9,
        margin=4,
        max_words=top_n,
        collocations=False,
        contour_color="#003366",
        contour_width=1,
        font_path="/System/Library/Fonts/Helvetica.ttc",
        color_func = frequency_color_func(freq_dict)
    ).generate_from_frequencies(freq_dict)

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    # --- Auto-generate title if not provided ---
    if title is None:
        title = infer_county_name(outpath)

    # --- Title styling ---
    if title:
        plt.suptitle(
            title,
            fontsize=20,
            fontweight="bold",
            color="#003366",
            y=1.03,
        )
    plt.title(
        f"Top {top_n} Most Frequent Words in Report",
        fontsize=14,
        color="#444444",
        style="italic",
        pad=12,
    )

    plt.tight_layout()

    # --- Output ---
    if show:
        plt.show()
    else:
        print(f"Saving wordcloud → {outpath}")
        plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------
# 6. THEMATIC TOKEN GROUPS (for Simple Reporting)
# ----------------------------------------------------

# Define keyword groups for thematic analysis
CHALLENGE_TERMS = {
    "barrier", "barriers", "challenge", "challenges", "transportation", "housing",
    "homeless", "homelessness", "childcare", "care", "domestic", "abuse", "violence",
    "mental", "health", "poverty", "cost", "costs", "wage", "wages", "language",
    "bilingual", "access", "disparities", "equity", "sanction", "sanctions", "sanctioned",
    "rent", "eviction", "evictions"
}

CONTEXT_TERMS = {
    "family", "families", "parent", "parents", "child", "children", "bilingual",
    "spanish", "english", "immigrant", "immigrants", "refugee", "refugees",
    "rural", "urban", "poverty", "wage", "wages", "income", "earnings",
    "participation", "engagement", "engage", "engaged", "work", "employment",
    "job", "jobs", "workforce"
}


def extract_policy_themes(tokens: list[str], keywords: set[str], top_n: int = 5) -> list[str]:
    """
    Extract the top-N most frequent policy-related keywords present in a token list.

    Args:
        tokens (list[str]): List of lemmatized tokens.
        keywords (set[str]): Set of thematic keywords to look for.
        top_n (int, optional): Number of top results to return (default = 5).

    Returns:
        list[str]: Top-N keyword strings by descending frequency.
    """
    freq = Counter(t for t in tokens if t in keywords)
    return [w for w, _ in freq.most_common(top_n)]

# ----------------------------------------------------
# 7. WORD EMBEDDINGS (Load pretrained → fallback local Word2Vec)
# ----------------------------------------------------

def build_sentences_for_w2v(raw_text: str) -> list[list[str]]:
    """
    Tokenize text into cleaned, lemmatized sentences for Word2Vec training or inference.

    Parameters
    ----------
    raw_text : str
        The full raw document text.

    Returns
    -------
    list[list[str]]
        List of token lists (one per sentence).
    """
    sentences = []
    for sent in nltk.sent_tokenize(raw_text):
        tokens = clean_and_tokenize(sent)
        if tokens:
            sentences.append(tokens)
    return sentences


def get_w2v_model(sentences: list[list[str]]):
    """
    Load a pretrained GloVe model (100D) if available, otherwise train a local Word2Vec model.

    Parameters
    ----------
    sentences : list[list[str]]
        Tokenized and cleaned sentences used for training if no pretrained model is found.

    Returns
    -------
    tuple
        (model, keyed_vectors), where either element may be None if loading/training fails.
    """
    if not W2V_OK:
        print("[WARN] gensim not available — Word2Vec disabled.")
        return None, None

    # --- 1) Try pretrained GloVe ---
    try:
        wv = gensim_load("glove-wiki-gigaword-100")
        print("[INFO] Loaded pretrained GloVe embeddings (100D).")
        return None, wv
    except Exception:
        print("[WARN] Could not load GloVe — training Word2Vec locally.")

    # --- 2) Fallback: train local Word2Vec ---
    try:
        model = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=3,
            workers=4,
            sg=1,
            negative=10,
            epochs=15,
        )
        print("[INFO] Trained local Word2Vec model on corpus.")
        return model, model.wv
    except Exception as e:
        print(f"[ERROR] Word2Vec training failed: {e}")
        return None, None


def nearest_terms(wv, term: str, topn: int = 10) -> list[str]:
    """
    Return the top-N nearest terms for a given word from a KeyedVectors model.

    Parameters
    ----------
    wv : gensim KeyedVectors
        The word embedding space.
    term : str
        Query word.
    topn : int
        Number of neighbors to return.

    Returns
    -------
    list[str]
        List of most similar words.
    """
    if (wv is None) or (not hasattr(wv, "key_to_index")):
        return []
    if term not in wv.key_to_index:
        return []
    try:
        return [w for w, _ in wv.most_similar(term, topn=topn)]
    except Exception:
        return []


def vector_mean(wv, tokens: list[str]) -> np.ndarray | None:
    """
    Compute the mean embedding vector for a list of tokens found in the embedding model.

    Parameters
    ----------
    wv : gensim KeyedVectors
        The embedding space.
    tokens : list[str]
        Tokens to average.

    Returns
    -------
    np.ndarray or None
        Mean vector, or None if no valid tokens are present.
    """
    if (wv is None) or (not hasattr(wv, "key_to_index")):
        return None
    vectors = [wv[t] for t in tokens if t in wv.key_to_index]
    return np.mean(vectors, axis=0) if vectors else None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors, with NaN safety.

    Parameters
    ----------
    a, b : np.ndarray
        Input vectors.

    Returns
    -------
    float
        Cosine similarity value, or np.nan if undefined.
    """
    if a is None or b is None:
        return np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


THEMATIC_SEEDS = {
    "employment": ["employment", "job", "workforce", "earnings"],
    "childcare": ["childcare", "child", "care", "parent"],
    "housing": ["housing", "rent", "homelessness", "shelter"],
    "transportation": ["transportation", "bus", "transit", "distance"],
    "mental_health": ["mental", "behavioral", "health", "counseling"],
    "sanctions": ["sanction", "compliance", "penalty", "good", "cause"],
    "language_access": ["language", "translation", "interpreter", "english", "spanish"],
    "violence": ["domestic", "violence", "abuse", "safety"],
    "equity": ["equity", "disparities", "racial"],
    "poverty": ["poverty", "lowincome", "income", "cost", "costs"],
}


def build_theme_vectors(wv) -> dict[str, np.ndarray | None]:
    """
    Construct average embedding vectors for each thematic seed list.

    Parameters
    ----------
    wv : gensim KeyedVectors
        The embedding space.

    Returns
    -------
    dict[str, np.ndarray or None]
        Mapping of theme → mean vector.
    """
    theme_vectors = {}
    for theme, seeds in THEMATIC_SEEDS.items():
        valid_seeds = [s for s in seeds if s in (wv.key_to_index if wv else {})]
        theme_vectors[theme] = vector_mean(wv, valid_seeds)
    return theme_vectors


def document_vector(wv, sentences: list[list[str]]) -> np.ndarray | None:
    """
    Compute a document-level mean embedding vector from all sentence tokens.

    Parameters
    ----------
    wv : gensim KeyedVectors
        The embedding space.
    sentences : list[list[str]]
        Tokenized, cleaned sentences.

    Returns
    -------
    np.ndarray or None
        Mean document vector.
    """
    flat_tokens = [t for sent in sentences for t in sent]
    return vector_mean(wv, flat_tokens)


def build_probe_prompt(county: str, theme_scores: dict[str, dict[str, float]], wv, top_k: int = 3) -> str:
    """
    Generate a short qualitative probing prompt based on top theme similarities.

    Parameters
    ----------
    county : str
        County label.
    theme_scores : dict
        Mapping of county → {theme: cosine similarity}.
    wv : gensim KeyedVectors
        Word embedding space.
    top_k : int
        Number of themes to display.

    Returns
    -------
    str
        Human-readable probing suggestion text.
    """
    if county not in theme_scores:
        return f"No theme scores found for {county}."

    scores = theme_scores[county]
    ranked = sorted(scores.items(), key=lambda x: (-x[1] if x[1] == x[1] else 1))
    top_themes = ranked[:top_k]

    bullet_lines = []
    for theme, score in top_themes:
        seeds = THEMATIC_SEEDS[theme]
        neighbors = []
        for s in seeds[:2]:
            neighbors += nearest_terms(wv, s, topn=5)
        seen = set()
        unique_neighbors = []
        for n in neighbors:
            if n not in seen:
                unique_neighbors.append(n)
                seen.add(n)
            if len(unique_neighbors) >= 6:
                break
        suggestion = ", ".join(unique_neighbors) if unique_neighbors else "related terms"
        bullet_lines.append(f"- {theme} (score={score:.2f}): consider {suggestion}")

    return f"Key areas to probe for {county}:\n" + "\n".join(bullet_lines)


# ----------------------------------------------------
# 8A. MAIN ANALYSIS (Top-20 WordClouds + Sentence-level Sentiment)
# ----------------------------------------------------

def analyze_reports(file_paths: list[Path], output_dir: Path) -> dict:
    """
    Analyze a collection of county PDF/DOCX reports.

    Steps:
    - Read and clean text
    - Compute token frequencies and top-20 WordClouds
    - Extract thematic keywords
    - Compute overall and per-sentence sentiment
    - Export per-county CSV summaries

    Parameters
    ----------
    file_paths : list[Path]
        List of input document paths.
    output_dir : Path
        Directory to save CSVs and visualizations.

    Returns
    -------
    dict
        Results by county (tokens, sentiment, stats, etc.).
    """
    sia = SentimentIntensityAnalyzer()
    results = {}
    statewide_tokens = []

    for path in file_paths:
        path = Path(path)
        county = infer_county_name(str(path))
        print(f"\n=== Processing {county} ===")

        # --- Text extraction ---
        raw_text = read_any_text(path)
        print(f"[INFO] Extracted {len(raw_text)} characters from {county}.")

        # --- Tokenization and normalization ---
        tokens = clean_and_tokenize(raw_text)
        tokens_collapsed = collapse_tokens_to_canon(tokens)
        statewide_tokens.extend(tokens_collapsed)

        # --- Sentiment analysis ---
        sent_score = compute_sentiment_sentence_avg(raw_text, sia)
        sent_label = interpret_sentiment(sent_score)

        # --- Thematic highlights ---
        top_challenges = extract_policy_themes(tokens, CHALLENGE_TERMS, top_n=5)
        top_context = extract_policy_themes(tokens, CONTEXT_TERMS, top_n=5)

        # --- Frequency analysis and WordCloud ---
        stats_df = compute_word_stats(tokens_collapsed)
        csv_out = output_dir / f"stats_{county}_collapsed.csv"
        stats_df.head(200).to_csv(csv_out, index=False)
        print(f"[INFO] Saved word frequency table → {csv_out.name}")

        wc_out = output_dir / f"wordcloud_{county}.png"
        build_wordcloud(
            stats_df,
            wc_out,
            title=f"{county} – Collapsed Top Terms (Top-20)",
            top_n=20,
        )

        # --- Compile results ---
        results[county] = {
            "tokens": tokens_collapsed,
            "sentiment_score": sent_score,
            "sentiment_label": sent_label,
            "top_challenges": top_challenges,
            "top_context": top_context,
            "freq_table": stats_df,
        }

    print("\nCompleted report analysis for all counties.")
    return results

# ----------------------------------------------------
# 8B. SENTIMENT ANALYSIS BLOCK (Separated)
#      - Sentence-level average sentiment
#      - Tone interpretation
#      - Overall sentiment bar plot
#      - Per-word sentiment estimation
# ----------------------------------------------------

def compute_sentiment_sentence_avg(raw_text: str, sia) -> float:
    """
    Compute the average VADER compound score across all valid sentences.

    Parameters
    ----------
    raw_text : str
        Full document text.
    sia : nltk.sentiment.SentimentIntensityAnalyzer
        Pre-initialized VADER sentiment analyzer.

    Returns
    -------
    float
        Mean compound sentiment score in [-1, 1].
        Returns 0.0 if no valid sentences are found.
    """
    sentences = nltk.sent_tokenize(raw_text)
    scores = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 5:  # Skip very short fragments
            continue
        scores.append(sia.polarity_scores(sent)["compound"])
    return (sum(scores) / len(scores)) if scores else 0.0


def interpret_sentiment(score: float) -> str:
    """
    Convert an average sentiment score into a qualitative label.

    Parameters
    ----------
    score : float
        VADER compound score (−1 to +1).

    Returns
    -------
    str
        Human-readable tone interpretation.
    """
    if score < 0.05:
        return "problem-focused tone (risks/gaps/barriers)"
    elif score < 0.15:
        return "mixed tone (barriers & solutions)"
    else:
        return "strength-oriented tone (successes/improvements)"


def plot_sentiment_bar(sentiment_avg: float, outpath: Path, title: str | None = None) -> None:
    """
    Save a simple vertical bar plot representing overall sentiment (–1..1).

    Parameters
    ----------
    sentiment_avg : float
        Average document sentiment.
    outpath : Path
        Output path for the PNG figure.
    title : str, optional
        Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")  # Ensure non-interactive backend

    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)
    ax.bar([0], [sentiment_avg], color="skyblue", edgecolor="black")
    ax.set_ylim(-1, 1)
    ax.set_xticks([0])
    ax.set_xticklabels(["sentiment"])
    ax.axhline(0, lw=1, color="gray")
    ax.set_ylabel("VADER compound (–1..1)")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    print(f"📁 Saving sentiment bar: {outpath}")
    plt.savefig(outpath, dpi=300)
    plt.close(fig)


def _sentence_tokens_canonical(sentence: str) -> set[str]:
    """
    Tokenize and normalize a sentence, collapsing tokens to canonical forms.

    Ensures consistent token treatment across the frequency table
    and sentiment computations.

    Parameters
    ----------
    sentence : str
        Raw sentence text.

    Returns
    -------
    set[str]
        Canonicalized tokens (unique within the sentence).
    """
    tokens = clean_and_tokenize(sentence)
    return set(collapse_tokens_to_canon(tokens))


def compute_word_level_sentiment(
    raw_text: str,
    freq_df: pd.DataFrame,
    top_n: int = 20,
    sia: SentimentIntensityAnalyzer | None = None
) -> pd.DataFrame:
    """
    Compute average sentence-level sentiment for each of the Top-N frequent words.

    Parameters
    ----------
    raw_text : str
        Full document text.
    freq_df : pd.DataFrame
        Frequency table (with 'word' and 'count' columns).
    top_n : int, default=20
        Number of top words to analyze.
    sia : SentimentIntensityAnalyzer, optional
        Existing analyzer instance; creates one if not provided.

    Returns
    -------
    pd.DataFrame
        Columns: ["word", "avg_sentiment", "n_sentences"].
    """
    if sia is None:
        sia = SentimentIntensityAnalyzer()

    if freq_df is None or freq_df.empty:
        return pd.DataFrame(columns=["word", "avg_sentiment", "n_sentences"])

    # --- Select Top-N canonical words ---
    top_words = list(freq_df.head(top_n)["word"].values)

    # --- Tokenize into sentences and precompute per-sentence data ---
    sentences = [s.strip() for s in nltk.sent_tokenize(raw_text) if len(s.strip()) >= 5]
    sent_token_sets = [_sentence_tokens_canonical(s) for s in sentences]
    sent_scores = [sia.polarity_scores(s)["compound"] for s in sentences]

    # --- Aggregate sentiment per word ---
    rows = []
    for w in top_words:
        relevant_scores = [
            sc for stoks, sc in zip(sent_token_sets, sent_scores) if w in stoks
        ]
        avg_sent = (sum(relevant_scores) / len(relevant_scores)) if relevant_scores else np.nan
        rows.append({"word": w, "avg_sentiment": avg_sent, "n_sentences": len(relevant_scores)})

    return pd.DataFrame(rows)


def plot_word_sentiment_bar(
    word_sent_df: pd.DataFrame,
    outpath: Path,
    title: str | None = None
) -> None:
    """
    Plot average sentence sentiment per Top-N word as a labeled bar chart.

    Parameters
    ----------
    word_sent_df : pd.DataFrame
        DataFrame with columns ["word", "avg_sentiment", "n_sentences"].
    outpath : Path
        Output path for saved PNG.
    title : str, optional
        Plot title.
    """
    if word_sent_df is None or word_sent_df.empty:
        print(f"[WARN] No per-word sentiment data available for {outpath}.")
        return

    # --- Clean and convert numeric fields ---
    plot_df = word_sent_df.copy()
    plot_df["avg_sentiment"] = pd.to_numeric(plot_df["avg_sentiment"], errors="coerce").fillna(0.0)
    plot_df["n_sentences"] = (
        pd.to_numeric(plot_df["n_sentences"], errors="coerce").fillna(0).astype(int)
    )

    x = plot_df["word"].values
    y = plot_df["avg_sentiment"].values
    counts = plot_df["n_sentences"].values

    if len(x) == 0:
        print(f"[WARN] No valid words to plot for {outpath}.")
        return

    # --- Non-interactive backend for safe file export ---
    import matplotlib
    matplotlib.use("Agg")

    # --- Build the figure ---
    fig = plt.figure(figsize=(max(8, len(x) * 0.6), 5))
    ax = fig.add_subplot(111)
    ax.bar(range(len(x)), y, color="skyblue", edgecolor="black")
    ax.set_ylim(-1, 1)
    ax.axhline(0, lw=1, color="gray")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Avg sentence sentiment (VADER)")
    ax.set_xlabel("Top frequent words")

    # --- Annotate with sentence counts ---
    for i, (val, n) in enumerate(zip(y, counts)):
        try:
            ax.text(
                i,
                float(val) + (0.02 if val >= 0 else -0.05),
                f"n={int(n)}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )
        except Exception:
            continue  # Safeguard against NaN values

    if title:
        ax.set_title(title)

    plt.tight_layout()
    print(f"📁 Saving per-word sentiment bar: {outpath}")
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

# ----------------------------------------------------
# 9. RUN SCRIPT (entry point)
# ----------------------------------------------------
if __name__ == "__main__":
    """
    Entry point for the analysis pipeline.
    Executes the full county-level + statewide report generation.
    """

    from pathlib import Path
    import os

    # --- Step 1. Prepare inputs ---
    try:
        FILE_PATHS = [str(p) for p in pdf_files]  # List of PDF files
    except NameError:
        raise RuntimeError(
            "Variable 'pdf_files' not defined. Ensure input PDF list is initialized before running."
        )

    # Convert OUTPUT_DIR to Path
    OUTPUT_DIR = Path(OUTPUT_DIR)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # safer than os.makedirs

    print("Launching full analysis pipeline...")
    print(f"Total files to analyze: {len(FILE_PATHS)}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # --- Step 2. Run main analysis ---
    results = analyze_reports(FILE_PATHS, OUTPUT_DIR)  # <-- pass the Path, not str()

    # --- Step 3. Print per-county summary ---
    print("==== County Summary (Collapsed Tokens) ====")
    for county, info in results.items():
        # Skip non-county entries
        if not isinstance(info, dict) or "sentiment_score" not in info:
            continue

        print(f"\n[{county}]")
        print(f"Sentiment: {info['sentiment_score']:.3f} → {info['sentiment_label']}")
        print("Top challenge themes:", ", ".join(info.get("top_challenges", [])) or "N/A")
        print("Top context themes:", ", ".join(info.get("top_context", [])) or "N/A")
        print("Top terms (collapsed):")
        freq_table = info.get("freq_table")
        if freq_table is not None and not freq_table.empty:
            for _, row in freq_table.head(5).iterrows():
                print(f" - {row['word']} (freq={row['freq']})")
        else:
            print(" - N/A")

    # --- Step 4. Statewide sentiment summary (optional) ---
    statewide_sentiments = [
        info['sentiment_score'] 
        for info in results.values()
        if isinstance(info, dict) and 'sentiment_score' in info
    ]
    if statewide_sentiments:
        avg_sentiment = sum(statewide_sentiments) / len(statewide_sentiments)
        print(f"\nStatewide sentiment (average across counties): {avg_sentiment:.3f}")
    else:
        print("\nStatewide sentiment not available.")

    # --- Step 5. Optional: Word2Vec theme scores ---
    theme_scores = results.get("theme_scores")
    if theme_scores:
        print("\n==== Theme Cosine Similarity (sample) ====")
        sample_county = next(iter(theme_scores.keys()))
        print(f"[{sample_county}] {theme_scores[sample_county]}")

    # --- Step 6. Optional: Generated prompts ---
    prompts = results.get("prompts")
    if prompts:
        print("\n==== Prompt Example ====")
        sample_county = next(iter(prompts.keys()))
        print(f"[{sample_county}]\n{prompts[sample_county]}")

    print("\nAnalysis complete. All outputs saved to:", OUTPUT_DIR)
