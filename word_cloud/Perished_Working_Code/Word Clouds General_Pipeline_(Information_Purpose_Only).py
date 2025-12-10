"""
This code is not meant to be run as-is. It serves as a reference, describing the current pipeline
to extract and analyze key policy terms from CalWORKs County Self-Assessments (CSAs), and generate 
word clouds and sentiment analyses.

CalWORKs CSA Policy Analysis Pipeline
-------------------------------------
Extracts and analyzes key policy terms from CalWORKs County Self-Assessments (CSAs),
including:
  1. Parsing and text cleaning
  2. Token frequency tables
  3. TF–IDF summaries
  4. Word cloud visualization
  5. Sentence-level sentiment using VADER
  6. Per-word sentiment aggregation
  7. Optional export to Excel/CSV

Author: [Your Name]
Date: 2025-11-08
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required data
nltk.download("vader_lexicon", quiet=True)

# ---------------------------------------------------------------
# Text cleaning & token utilities
# ---------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, stopwords: set) -> list[str]:
    return [w for w in text.split() if w not in stopwords and len(w) > 2]


# ---------------------------------------------------------------
# Token frequency table
# ---------------------------------------------------------------
def build_token_table(texts: dict[str, str], stopwords: set) -> pd.DataFrame:
    records = []
    for county, txt in texts.items():
        tokens = tokenize(clean_text(txt), stopwords)
        for t in tokens:
            records.append({"county": county, "token": t})
    df = pd.DataFrame(records)
    freq = df.groupby(["county", "token"]).size().reset_index(name="count")
    return freq


# ---------------------------------------------------------------
# TF-IDF calculation
# ---------------------------------------------------------------
def compute_tfidf(freq: pd.DataFrame) -> pd.DataFrame:
    """Compute a basic TF-IDF weighting by county."""
    df = freq.copy()
    total_tokens = df.groupby("county")["count"].transform("sum")
    df["tf"] = df["count"] / total_tokens
    doc_freq = df.groupby("token")["county"].nunique()
    n_docs = df["county"].nunique()
    df["idf"] = df["token"].map(lambda w: np.log((1 + n_docs) / (1 + doc_freq[w])) + 1)
    df["tfidf"] = df["tf"] * df["idf"]
    return df


# ---------------------------------------------------------------
# Word cloud plotting
# ---------------------------------------------------------------
def plot_wordcloud(freq: pd.DataFrame, county: str, outdir: str, top_n: int = 20):
    """Generate and save a word cloud for a given county."""
    subset = freq[freq["county"] == county].nlargest(top_n, "count")
    word_freq = dict(zip(subset["token"], subset["count"]))
    if not word_freq:
        print(f"⚠️ No tokens found for {county}.")
        return
    wc = WordCloud(
        width=800, height=400, background_color="white",
        colormap="viridis", max_words=top_n
    ).generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Top {top_n} Words – {county}")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"wordcloud_{county}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"✅ Saved word cloud: {outpath}")


# ---------------------------------------------------------------
# Sentence-level sentiment
# ---------------------------------------------------------------
def compute_sentence_sentiment(texts: dict[str, str]) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    records = []
    for county, txt in texts.items():
        sentences = re.split(r"(?<=[.!?]) +", txt)
        for sent in sentences:
            if not sent.strip():
                continue
            score = sia.polarity_scores(sent)["compound"]
            records.append({"county": county, "sentence": sent, "sentiment": score})
    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------
# Word-level sentiment aggregation
# ---------------------------------------------------------------
def word_level_sentiment(texts: dict[str, str], stopwords: set) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    records = []
    for county, txt in texts.items():
        sentences = re.split(r"(?<=[.!?]) +", txt)
        for sent in sentences:
            score = sia.polarity_scores(sent)["compound"]
            tokens = tokenize(clean_text(sent), stopwords)
            for t in tokens:
                records.append({"county": county, "word": t, "sentiment": score})
    df = pd.DataFrame(records)
    return (
        df.groupby(["county", "word"])["sentiment"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_sentiment", "count": "n_sentences"})
    )


# ---------------------------------------------------------------
# Sentiment visualization
# ---------------------------------------------------------------
def plot_word_sentiment_bar(df: pd.DataFrame, county: str, outdir: str, top_n: int = 20):
    """Bar plot of average sentiment for the most frequent words per county."""
    subset = df[df["county"] == county]
    if subset.empty:
        print(f"⚠️ No sentiment data for {county}")
        return
    top_words = subset.nlargest(top_n, "n_sentences")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_words["word"], top_words["avg_sentiment"], color="skyblue", edgecolor="black")
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("Average VADER Sentiment (–1 to +1)")
    ax.set_ylabel("Word")
    ax.set_title(f"Word Sentiment – {county}")
    for i, (val, n) in enumerate(zip(top_words["avg_sentiment"], top_words["n_sentences"])):
        ax.text(val, i, f" ({n})", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"sentiment_bar_{county}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"✅ Saved sentiment bar: {outpath}")


# ---------------------------------------------------------------
# Example run (for debugging/demo)
# ---------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    sample_texts = {
        "Alameda": "CalWORKs helps families find employment. Families are supported with childcare and education.",
        "Fresno": "The program provides cash aid and work opportunities. Families can gain stability and independence.",
    }

    STOPWORDS = {
        "the", "and", "of", "in", "to", "a", "for", "on", "is",
        "are", "with", "as", "by", "an", "be", "that", "this",
        "at", "from", "was", "were", "it", "or", "we", "will",
        "our", "their", "can", "has", "have", "had"
    }

    # --- Run pipeline ---
    freq_table = build_token_table(sample_texts, STOPWORDS)
    print("Token frequency table:")
    print(freq_table.head(), "\n")

    tfidf_table = compute_tfidf(freq_table)
    print("TF–IDF table:")
    print(tfidf_table.head(), "\n")

    plot_wordcloud(freq_table, "Alameda", "outputs")
    sentiment_df = compute_sentence_sentiment(sample_texts)
    print("Sentence-level sentiment:")
    print(sentiment_df.head(), "\n")

    word_sent_df = word_level_sentiment(sample_texts, STOPWORDS)
    print("Word-level sentiment:")
    print(word_sent_df.head(), "\n")

    plot_word_sentiment_bar(word_sent_df, "Alameda", "outputs")
