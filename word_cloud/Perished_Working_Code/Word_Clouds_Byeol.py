########################################
# CalWORKs CSA Policy Analysis Pipeline
# - Robust PDF extraction (pdfplumber -> PyPDF2 fallback)
# - Preserve original stopwords & KEEP_TERMS (same content/logic)
# - Defensive compute_word_stats (no KeyError on empty)
# - No shadowing of pathlib.Path
# - Clear debug logs
# - WordCloud shows TOP-20 terms only
# - Sentiment block separated
# - NEW: For each county, compute sentence-level sentiment per Top-20 word
#        (average sentiment of sentences that include the word) and plot a bar chart
########################################

import os
import re
import zipfile
from pathlib import Path as _Path
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

# ---- Quiet noisy PDF warnings (pdfminer/ghostscript style) ----
import logging
for noisy in ("pdfminer", "pdfminer.pdfinterp", "pdfminer.cmapdb", "pdfminer.psparser", "pdfminer.pdfpage"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


# Optional: docx readers (kept for completeness)
import docx2txt
from docx import Document

# ==== Optional: gensim (Word2Vec) ====
W2V_OK = True
try:
    from gensim.models import Word2Vec, KeyedVectors
    from gensim.downloader import load as gensim_load
except Exception:
    W2V_OK = False

# ==== Optional: PDF readers ====
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

# ----------------------------------------------------
# 0. NLTK resources
# ----------------------------------------------------
def _safe_download(resource_path, download_name):
    """Ensure an NLTK resource exists; download if missing."""
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name)

_safe_download("tokenizers/punkt", "punkt")
_safe_download("tokenizers/punkt_tab", "punkt_tab")
_safe_download("corpora/stopwords", "stopwords")
_safe_download("sentiment/vader_lexicon", "vader_lexicon")

# ----------------------------------------------------
# 1. USER CONFIG
# ----------------------------------------------------
# ZIP → 'CalWORKs data_extracted' folder
ZIP_PATH = r'/Users/cyrillefougere/Desktop/CalWORKs data.zip'  # edit if needed
BASE_DIR = _Path(ZIP_PATH).parent
EXTRACT_DIR = BASE_DIR / "CalWORKs data_extracted"
os.makedirs(EXTRACT_DIR, exist_ok=True)

print("=== Vérification des chemins ===")
print("ZIP exists:", _Path(ZIP_PATH).exists())
print("Extract dir:", EXTRACT_DIR)
print("Output dir:", OUTPUT_DIR)
print("===============================")

# 1) Extract ZIP once
if (not any(EXTRACT_DIR.rglob("*"))) and _Path(ZIP_PATH).exists():
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
print(f"📦 Extracted ZIP to: {EXTRACT_DIR}")

# 2) Analyze PDFs only
pdf_files = list(EXTRACT_DIR.rglob("*.pdf"))
print("🔍 Files to analyze:")
for f in pdf_files:
    print(" -", f.relative_to(EXTRACT_DIR))

OUTPUT_DIR = BASE_DIR / "Word Clouds Update"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n📂 Output: {OUTPUT_DIR}\n")

# ----------------------------------------------------
# 2. TEXT READER (PDF/ DOCX)
# ----------------------------------------------------
def read_pdf_text(path: _Path) -> str:
    # ---- Optional: PyMuPDF (fitz) and OCR fallback ----
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

def _ocr_pdf_pages_to_text(path, max_pages=5):
    """Lightweight OCR fallback: rasterize first `max_pages` and OCR them."""
    if not _HAS_OCR or not _HAS_FITZ:
        return ""
    text_parts = []
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
    return "\n".join(text_parts)

def read_pdf_text(path: _Path) -> str:
    """Try pdfplumber -> PyMuPDF (fitz) -> PyPDF2 -> OCR (subset) -> ''."""
    # 1) pdfplumber
    if _PDF_BACKENDS.get("pdfplumber", False):
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                parts = []
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t.strip():
                        parts.append(t)
                if parts:
                    text = "\n".join(parts)
                    print(f"[PDF] {path.name} → pdfplumber OK ({len(text)} chars)")
                    return text
                else:
                    print(f"[PDF] {path.name} → pdfplumber empty.")
        except Exception as e:
            print(f"[PDF] pdfplumber failed on {path.name}: {e}")

    # 2) PyMuPDF (fitz)
    if _HAS_FITZ:
        try:
            doc = fitz.open(str(path))
            parts = []
            for page in doc:
                t = page.get_text("text") or ""
                if t.strip():
                    parts.append(t)
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → PyMuPDF OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → PyMuPDF empty.")
        except Exception as e:
            print(f"[PDF] PyMuPDF failed on {path.name}: {e}")

    # 3) PyPDF2
    if _PDF_BACKENDS.get("pypdf2", False):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(path))
            parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → PyPDF2 OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → PyPDF2 empty.")
        except Exception as e:
            print(f"[PDF] PyPDF2 failed on {path.name}: {e}")

    # 4) OCR fallback (sample pages only)
    ocr_text = _ocr_pdf_pages_to_text(path, max_pages=5)
    if ocr_text.strip():
        print(f"[PDF] {path.name} → OCR sample OK ({len(ocr_text)} chars; first 5 pages)")
        return ocr_text

    print(f"[PDF] {path.name} → NO TEXT (scanned/secured?)")
    return ""

def read_docx_text(path: _Path) -> str:
    """Best-effort DOCX reader (python-docx → docx2txt fallback)."""
    text1 = ""
    try:
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs]
        text1 = "\n".join(paragraphs)
    except Exception:
        text1 = ""
    if len(text1.strip()) < 50:
        try:
            text2 = docx2txt.process(str(path))
            if len(text2.strip()) > len(text1.strip()):
                return text2
        except Exception:
            pass
    return text1

def read_any_text(path: _Path) -> str:
    """Dispatch by extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf_text(path)
    elif ext == ".docx":
        return read_docx_text(path)
    else:
        return ""

# ----------------------------------------------------
# 3. STOPWORDS (policy-tuned) + GEO TERMS
# ----------------------------------------------------
def build_stopwords():
    base_sw = set(stopwords.words('english'))

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

    org_terms = {
    "agencies", "agency", "caseworker", "caseworkers", "cbos", "cdss", "cct", "center", "centers",
    "client", "clients", "collaborator", "collaborators", "customer", "customers",
    "calsaw", "calsaws", "calworks", "cwd", "debs", "department", "departments",
    "des", "dess", "dha", "doe", "dss", "ecm", "ecms", "edc", "ess", "esss", "etw", "hhs",
    "hsa", "icdss", "kchsa", "manager", "managers", "mcdss", "oar", "ocat", "office", "offices",
    "ore", "ota", "partner", "partners", "participant", "participants", "programmatic",
    "provider", "providers", "recipient", "recipients", "sfhsa", "specialist", "specialists",
    "staff", "staffed", "staffing", "system", "tad", "team", "teams", "ts", "unit", "units",
    "vendor", "vendors", "vchsa", "ychhsd" "welfare", "wex"
    }   


    geo_terms = {
    # generic geo
    "area", "areas", "bay", "cal", "california", "central", "city", "cities",
    "coast", "coastal", "county", "counties", "district", "districts", "inland",
    "metro", "northern", "region", "regional", "southern", "valley", "valleys",
    # common tokens in county names
    "barbara", "benito", "bernardino", "clara", "contra", "costa", "cruz", "de", "del",
    "dorado", "diego", "el", "francisco", "glenn", "humboldt", "imperial", "inyo",
    "joaquin", "kern", "kings", "lake", "lassen", "la", "luis", "marin", "mariposa",
    "mateo", "mendocino", "merced", "mono", "monterey", "napa", "nevada", "obispo",
    "orange", "placer", "plumas", "riverside", "sacramento", "san", "santa", "shasta",
    "sierra", "siskiyou", "solano", "sonoma", "stanislaus", "sutter", "tehama", "trinity",
    "tulare", "tuolumne", "ventura", "yolo", "yuba", "alameda", "alpine", "amador",
    "angeles", "butte", "calaveras", "colusa", "fresno", "kings", "madera", "modoc", "mono",
    "norte", "los", "angeles", "santa", "Santa"
    }


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
    "well", "yes", "year", "yearly", "years", "one", "toward", "available", "address",
    "section", "social", "high", "individual", "need", "etc", "rate"
    }

    
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

    sw = base_sw.union(admin_metrics).union(org_terms).union(geo_terms).union(report_filler)
    sw = {w for w in sw if w not in KEEP_TERMS}
    return sw

STOPWORDS = build_stopwords()

from nltk.stem import WordNetLemmatizer

# download WordNet if not already
_safe_download("corpora/wordnet", "wordnet")
_safe_download("corpora/omw-1.4", "omw-1.4")

lemmatizer = WordNetLemmatizer()

LEM = WordNetLemmatizer()

def lemmatize_token(t: str) -> str:
    # noun → verb → adj 
    t1 = LEM.lemmatize(t, pos='n')
    t2 = LEM.lemmatize(t1, pos='v')
    t3 = LEM.lemmatize(t2, pos='a')
    return t3

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    # NEW: lemmatize before stopword filtering
    lemmas = [lemmatize_token(t) for t in tokens]
    cleaned = [t for t in lemmas if t not in STOPWORDS and len(t) > 2 and not t.isnumeric()]
    return cleaned

# ----------------------------------------------------
# 4. SYNONYM GROUPING (thesaurus collapse)
# ----------------------------------------------------
def build_thesaurus():
    """Map semantically similar variants to a canonical token for counting/plotting."""
    groups = {
        "employment": {"employment","employ","employed","job","jobs","work","workforce"},
        "wages": {"wage","wages","pay","earnings","income"},
        "engagement": {"engagement","engage","engaged","participation","participate","outreach","attendance","orientation"},
        "sanctions": {"sanction","sanctions","sanctioned","compliance","penalty","penalties","noncompliance","good","cause","goodcause"},
        "housing": {"housing","rent","rents","eviction","evictions","homeless","homelessness","shelter"},
        "childcare": {"childcare","child","children","care","parent","parents","parenting","caregiver"},
        "transportation": {"transportation","bus","buses","transit","distance","commute"},
        "mental_health": {"mental","health","healthcare","behavioral","counseling","therapy"},
        "language_access": {"language","bilingual","spanish","english","interpreter","translation","access","accessibility"},
        "equity": {"equity","equitable","disparities","racial","race","racism"},
        "poverty": {"poverty","lowincome","low","income","cost","costs"},
        "violence": {"domestic","violence","abuse","dv","safety"},
        "technology": {"technology","digital","internet","online","device","devices"},
        "immigration": {"immigrant","immigrants","refugee","refugees"},
    }
    v2c = {}
    for canon, variants in groups.items():
        for v in variants:
            v2c[v] = canon
    return groups, v2c

THESAURUS, VAR2CANON = build_thesaurus()

def collapse_tokens_to_canon(tokens):
    """Replace each token by its canonical representative if available."""
    return [VAR2CANON.get(t, t) for t in tokens]

# ----------------------------------------------------
# 5. WORD STATS & PLOTS (WordCloud Top-20)
# ----------------------------------------------------
def compute_word_stats(tokens):
    """Return DataFrame[word, freq, relative_freq]; safe on empty."""
    total = len(tokens)
    if total == 0:
        return pd.DataFrame(columns=["word", "freq", "relative_freq"])
    freq = Counter(tokens)
    rows = [{"word": w, "freq": c, "relative_freq": (c/total)} for w,c in freq.items()]
    df = pd.DataFrame(rows).sort_values("freq", ascending=False).reset_index(drop=True)
    return df

def build_wordcloud(freq_df, outpath, title=None, top_n=20):
    """WordCloud: show only the top_n words (default 20)."""
    if freq_df.empty:
        print(f"[WARN] No words to plot for {outpath}")
        return
    freq_df_top = freq_df.head(top_n)
    freq_dict = dict(zip(freq_df_top["word"], freq_df_top["freq"]))
    if not freq_dict:
        print(f"[WARN] No words to plot for {outpath}")
        return
    wc = WordCloud(width=1000, height=800, background_color="white", max_words=top_n)\
         .generate_from_frequencies(freq_dict)
    fig = plt.figure(figsize=(10,7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title: plt.title(title)
    plt.tight_layout()
    print("📁 Saving wordcloud:", outpath)
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

def infer_county_name(path):
    """Create a compact label from filename."""
    base = os.path.basename(path)
    base_noext = os.path.splitext(base)[0]
    label = base_noext.replace("_"," ").strip()
    words = label.split()
    if len(words)>7: label = " ".join(words[:7])
    return label

# ----------------------------------------------------
# 6. THEMES (for simple reporting)
# ----------------------------------------------------
CHALLENGE_TERMS = [
    "barrier","barriers","challenge","challenges","transportation","housing",
    "homeless","homelessness","childcare","care","domestic","abuse","violence",
    "mental","health","poverty","cost","costs","wage","wages","language",
    "bilingual","access","disparities","equity","sanction","sanctions","sanctioned",
    "rent","eviction","evictions"
]
CONTEXT_TERMS = [
    "family","families","parent","parents","child","children","bilingual","spanish","english",
    "immigrant","immigrants","refugee","refugees","rural","urban","poverty","wage","wages",
    "income","earnings","participation","engagement","engage","engaged","work","employment",
    "job","jobs","workforce"
]
def extract_policy_themes(tokens, keywords, top_n=5):
    """Return top_n keywords present in tokens by frequency."""
    freq = Counter([t for t in tokens if t in keywords])
    return [w for w,_ in freq.most_common(top_n)]

# ----------------------------------------------------
# 7. WORD2VEC (load pretrained → fallback train)
# ----------------------------------------------------
def build_sentences_for_w2v(raw_text):
    """Tokenize sentences, clean tokens, return list of token lists (for w2v)."""
    out = []
    for s in nltk.sent_tokenize(raw_text):
        toks = clean_and_tokenize(s)
        if toks:
            out.append(toks)
    return out

def get_w2v_model(all_sentences):
    """Try GloVe vectors; else train Word2Vec on provided sentences; else None."""
    if not W2V_OK:
        return None, None
    try:
        wv = gensim_load("glove-wiki-gigaword-100")  # 100d
        return None, wv  # (model, keyedvectors)
    except Exception:
        pass
    try:
        model = Word2Vec(
            sentences=all_sentences,
            vector_size=100, window=5, min_count=3, workers=4, sg=1, negative=10, epochs=15
        )
        return model, model.wv
    except Exception:
        return None, None

def nearest_terms(wv, term, topn=10):
    """Return nearest neighbors for 'term' if available."""
    if (wv is None) or (not hasattr(wv, "key_to_index")): return []
    if term in wv.key_to_index:
        return [w for w,_ in wv.most_similar(term, topn=topn)]
    return []

def vec_mean(wv, tokens):
    """Mean vector of tokens present in wv."""
    if (wv is None) or (not hasattr(wv,"key_to_index")): return None
    vecs = [wv[t] for t in tokens if t in wv.key_to_index]
    return np.mean(vecs, axis=0) if len(vecs) else None

def cos_sim(a,b):
    """Cosine similarity with NaN safety."""
    if a is None or b is None: return np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na==0 or nb==0: return np.nan
    return float(np.dot(a,b)/(na*nb))

THEMES = {
    "employment": ["employment","job","workforce","earnings"],
    "childcare": ["childcare","child","care","parent"],
    "housing": ["housing","rent","homelessness","shelter"],
    "transport": ["transportation","bus","transit","distance"],
    "mental_health": ["mental","behavioral","health","counseling"],
    "sanctions": ["sanction","compliance","penalty","good","cause"],
    "language_access": ["language","translation","interpreter","english","spanish"],
    "dv": ["domestic","violence","abuse","safety"],
    "equity": ["equity","disparities","racial"],
    "poverty": ["poverty","lowincome","income","cost","costs"]
}

def theme_vectors(wv):
    """Build average vectors for theme seed lists."""
    tv = {}
    for k, seeds in THEMES.items():
        v = vec_mean(wv, [s for s in seeds if s in (wv.key_to_index if wv else {})])
        tv[k] = v
    return tv

def county_doc_vector(wv, token_lists):
    """Document vector = mean of all sentence tokens that exist in wv."""
    flat = []
    for lst in token_lists:
        flat.extend(lst)
    return vec_mean(wv, flat)

def build_prompt(county, theme_scores, wv, top_k=3):
    """Create a short probing prompt from top theme scores and nearest neighbors."""
    scores = theme_scores[county]
    ranked = sorted([(t, s) for t,s in scores.items()],
                    key=lambda x: (-x[1] if (x[1]==x[1]) else 1))
    top = ranked[:top_k]
    bullets = []
    for theme, sc in top:
        seeds = THEMES[theme]
        nn = []
        for s in seeds[:2]:
            nn += nearest_terms(wv, s, topn=5)
        seen = set(); nn2=[]
        for w in nn:
            if w not in seen:
                nn2.append(w); seen.add(w)
            if len(nn2)>=6: break
        bullets.append(f"- {theme} (score={sc:.2f}): consider {', '.join(nn2) if nn2 else 'domain details'}")
    return "Key areas to probe:\n" + "\n".join(bullets)

# ----------------------------------------------------
# 8A. MAIN ANALYSIS (WordCloud Top-20; overall sentiment saved separately)
#     + NEW per-word sentence sentiment for Top-20 words
# ----------------------------------------------------
def analyze_reports(file_paths, output_dir):
    sia = SentimentIntensityAnalyzer()
    county_results = {}
    statewide_tokens_all = []
    statewide_texts_all = []
    county_sent_tokens = {}

    for path in file_paths:
        path = _Path(path)
        county_label = infer_county_name(str(path))
        raw_text = read_any_text(path)
        print(f"[DEBUG] {county_label} length:", len(raw_text))

        # Tokenize/normalize and collapse to canonical
        toks = clean_and_tokenize(raw_text)
        toks_collapsed = collapse_tokens_to_canon(toks)

        # Overall sentiment (plotted by sentiment block)
        sent_score = compute_sentiment_sentence_avg(raw_text, sia)
        sent_label = interpret_sentiment(sent_score)

        # Simple theme snippets for the console summary
        top_challenges = extract_policy_themes(toks, CHALLENGE_TERMS, top_n=5)
        top_context   = extract_policy_themes(toks, CONTEXT_TERMS,   top_n=5)

        # Frequencies and WordCloud (Top-20)
        stats_df = compute_word_stats(toks_collapsed)
        csv_path = os.path.join(output_dir, f"stats_{county_label}_collapsed.csv")
        stats_df.head(200).to_csv(csv_path, index=False)

        wc_path = os.path.join(output_dir, f"wordcloud_{county_label}.png")
        build_wordcloud(
            stats_df,
            wc_path,
            title=f"{county_label} – collapsed top terms (Top-20)",
            top_n=20
        )

        # NEW: word-level sentence sentiment for Top-20 words
        word_sent_df = compute_word_level_sentiment(raw_text, stats_df, top_n=20, sia=sia)
        ws_csv = os.path.join(output_dir, f"word_sentiment_{county_label}.csv")
        word_sent_df.to_csv(ws_csv, index=False)

        ws_png = os.path.join(output_dir, f"word_sentiment_{county_label}.png")
        plot_word_sentiment_bar(
            word_sent_df,
            ws_png,
            title=f"{county_label} – Avg sentence sentiment per Top-20 word (VADER)"
        )

        # Save overall sentiment bar (separate simple plot)
        sent_img = os.path.join(output_dir, f"sentiment_{county_label}.png")
        plot_sentiment_bar(sent_score, sent_img, title=f"{county_label} – Overall tone (VADER)")

        # Accumulate for statewide summary
        statewide_tokens_all.extend(toks_collapsed)
        statewide_texts_all.append(raw_text)
        county_sent_tokens[county_label] = build_sentences_for_w2v(raw_text)

        county_results[county_label] = {
            "sentiment_score": sent_score,
            "sentiment_label": sent_label,
            "challenge_themes": top_challenges,
            "context_themes": top_context,
            "top_terms": stats_df.head(10).to_dict(orient="records"),
            "word_sentiment": word_sent_df.to_dict(orient="records")
        }

    # Statewide aggregate
    statewide_stats = compute_word_stats(statewide_tokens_all)
    statewide_csv = os.path.join(output_dir, "stats_California_statewide_collapsed.csv")
    statewide_stats.head(200).to_csv(statewide_csv, index=False)

    wc_state = os.path.join(output_dir, "wordcloud_California_statewide.png")
    build_wordcloud(
        statewide_stats,
        wc_state,
        title="California statewide – collapsed top terms (Top-20)",
        top_n=20
    )

    statewide_sent = compute_sentiment_sentence_avg("\n".join(statewide_texts_all), sia)
    wc_state_sent = os.path.join(output_dir, "sentiment_California_statewide.png")
    plot_sentiment_bar(statewide_sent, wc_state_sent, title="California statewide – Overall tone (VADER)")

    # Word2Vec (optional)
    theme_scores = {}
    prompts = {}
    if W2V_OK:
        all_sentences = []
        for c, sents in county_sent_tokens.items():
            all_sentences += sents

        model, wv = get_w2v_model(all_sentences)
        if wv is not None:
            tvecs = theme_vectors(wv)
            cvecs = {c: county_doc_vector(wv, sents) for c,sents in county_sent_tokens.items()}

            for c, cvec in cvecs.items():
                scores = {t: cos_sim(cvec, tvecs[t]) for t in THEMES.keys()}
                theme_scores[c] = scores

            if theme_scores:
                df_theme = pd.DataFrame(theme_scores).T
                df_theme.to_csv(os.path.join(output_dir, "theme_cosine_by_county.csv"))

            for c in county_sent_tokens.keys():
                prompts[c] = build_prompt(c, theme_scores, wv, top_k=3)
                with open(os.path.join(output_dir, f"prompt_{c}.txt"), "w", encoding="utf-8") as f:
                    f.write(prompts[c])
        else:
            print("⚠️ Word2Vec unavailable (gensim model not built/loaded). Skipping theme scores & prompts.")
    else:
        print("⚠️ gensim not installed. Skipping Word2Vec/theme scoring/prompt generation.")

    return {
        "counties": county_results,
        "statewide_sentiment": statewide_sent,
        "theme_scores": theme_scores if theme_scores else None,
        "prompts": prompts if prompts else None
    }

# ----------------------------------------------------
# 8B. SENTIMENT BLOCK (separated)
#     - sentence-level average
#     - labeling
#     - simple bar plot
#     - NEW helpers for per-word sentence sentiment
# ----------------------------------------------------
def compute_sentiment_sentence_avg(raw_text, sia):
    """Average VADER compound over sentences (skip very short fragments)."""
    sents = nltk.sent_tokenize(raw_text)
    scores = []
    for s in sents:
        s = s.strip()
        if len(s) < 5:
            continue
        scores.append(sia.polarity_scores(s)['compound'])
    return (sum(scores)/len(scores)) if scores else 0.0

def interpret_sentiment(score):
    """Heuristic interpretation for overall document tone."""
    if score < 0.05:
        return "problem-focused tone (risks/gaps/barriers)"
    elif score < 0.15:
        return "mixed tone (barriers & solutions)"
    else:
        return "strength-oriented tone (successes/improvements)"

def plot_sentiment_bar(sentiment_avg, outpath, title=None):
    import matplotlib
    matplotlib.use("Agg")
    """Save a simple bar showing overall sentiment (–1..1)."""
    fig = plt.figure(figsize=(4,5))
    ax = fig.add_subplot(111)
    ax.bar([0], [sentiment_avg])
    ax.set_ylim(-1, 1)
    ax.set_xticks([0]); ax.set_xticklabels(["sentiment"])
    ax.axhline(0, lw=1)
    ax.set_ylabel("VADER compound (-1..1)")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    print("📁 Saving sentiment bar:", outpath)
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

def _sentence_tokens_canonical(sentence):
    """
    Clean and tokenize a sentence using the same pipeline as documents
    and collapse to canonical tokens so matching is consistent with freq table.
    """
    toks = clean_and_tokenize(sentence)
    return set(collapse_tokens_to_canon(toks))

def compute_word_level_sentiment(raw_text, freq_df, top_n=20, sia=None):
    """
    For the Top-N words (by collapsed frequency), compute the average
    VADER sentiment of sentences that contain each word.
    Returns DataFrame[word, avg_sentiment, n_sentences].
    """
    if sia is None:
        sia = SentimentIntensityAnalyzer()

    if freq_df is None or freq_df.empty:
        return pd.DataFrame(columns=["word", "avg_sentiment", "n_sentences"])

    # Take Top-N canonical words
    top_words = list(freq_df.head(top_n)["word"].values)

    # Tokenize source text into sentences once, and precompute per-sentence sets
    sentences = [s.strip() for s in nltk.sent_tokenize(raw_text) if len(s.strip()) >= 5]
    sent_tok_sets = [_sentence_tokens_canonical(s) for s in sentences]
    sent_scores = [sia.polarity_scores(s)['compound'] for s in sentences]

    rows = []
    for w in top_words:
        # Collect scores of sentences whose canonical token set includes w
        scores_w = [sc for stoks, sc in zip(sent_tok_sets, sent_scores) if w in stoks]
        avg = (sum(scores_w)/len(scores_w)) if scores_w else np.nan
        rows.append({"word": w, "avg_sentiment": avg, "n_sentences": len(scores_w)})

    df = pd.DataFrame(rows)
    # Order by word frequency order (already in top_words order); ensure stable index
    return df

def plot_word_sentiment_bar(word_sent_df, outpath, title=None):
    """
    Plot average sentence sentiment per Top-20 word.
    Prevents NaN/invalid float errors when exporting as PNG/PDF.
    """
    if word_sent_df is None or word_sent_df.empty:
        print(f"[WARN] No per-word sentiment to plot for {outpath}")
        return

    plot_df = word_sent_df.copy()
    plot_df["avg_sentiment"] = pd.to_numeric(plot_df["avg_sentiment"], errors="coerce").fillna(0.0)
    plot_df["n_sentences"] = pd.to_numeric(plot_df["n_sentences"], errors="coerce").fillna(0).astype(int)

    x = plot_df["word"].values
    y = plot_df["avg_sentiment"].values
    counts = plot_df["n_sentences"].values

    if len(x) == 0:
        print(f"[WARN] No per-word sentiment to plot for {outpath}")
        return

    # Force non-interactive backend (avoids ghostscript float issues)
    import matplotlib
    matplotlib.use("Agg")

    fig = plt.figure(figsize=(max(8, len(x)*0.6), 5))
    ax = fig.add_subplot(111)
    ax.bar(range(len(x)), y, color="skyblue", edgecolor="black")
    ax.set_ylim(-1, 1)
    ax.axhline(0, lw=1, color="gray")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Avg sentence sentiment (VADER)")
    ax.set_xlabel("Top-20 frequent words")

    # Annotate with number of sentences used per word
    for i, (yy, n) in enumerate(zip(y, counts)):
        try:
            ax.text(i, float(yy) + (0.02 if yy >= 0 else -0.05),
                    f"n={int(n)}", ha="center",
                    va="bottom" if yy >= 0 else "top",
                    fontsize=8)
        except Exception:
            # Skip invalid text (NaN or non-float)
            continue

    if title:
        ax.set_title(title)
    plt.tight_layout()
    print("📁 Saving per-word sentiment bar:", outpath)
    plt.savefig(outpath, dpi=300)
    plt.close(fig)


# ----------------------------------------------------
# 9. RUN
# ----------------------------------------------------
if __name__ == "__main__":
    # pdf_files already contains absolute paths
    FILE_PATHS = [str(p) for p in pdf_files]
    results = analyze_reports(FILE_PATHS, str(OUTPUT_DIR))

    print("==== County summary (collapsed tokens) ====")
    for county, info in results["counties"].items():
        print(f"\n[{county}]")
        print(f"Sentiment: {info['sentiment_score']:.3f}  -> {info['sentiment_label']}")
        print("Top challenge themes:", ", ".join(info["challenge_themes"]) if info["challenge_themes"] else "N/A")
        print("Top context themes:  ", ", ".join(info["context_themes"]) if info["context_themes"] else "N/A")
        print("Top terms (collapsed):")
        for row in info["top_terms"][:5]:
            print(f" - {row['word']} (freq={row['freq']})")

    print("\nStatewide sentiment (collapsed):", f"{results['statewide_sentiment']:.3f}")

    if results["theme_scores"] is not None:
        print("\n==== Theme cosine (sample) ====")
        any_c = next(iter(results["theme_scores"].keys()))
        print(any_c, results["theme_scores"][any_c])

    if results["prompts"] is not None:
        print("\n==== Prompt example ====")
        any_c = next(iter(results["prompts"].keys()))
        print(f"[{any_c}]\n{results['prompts'][any_c]}")
