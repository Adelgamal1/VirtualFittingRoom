"""Analyze large text file sentiment with TextBlob."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from textblob import TextBlob


def classify_sentiment(polarity: float, threshold: float = 0.05) -> str:
    """Convert TextBlob polarity into positive, neutral, or negative."""
    if polarity > threshold:
        return "positive"
    if polarity < -threshold:
        return "negative"
    return "neutral"


def iter_text_chunks(file_path: Path, chunk_size: int = 5000):
    """Yield text chunks without loading a large file fully into memory."""
    with file_path.open("r", encoding="utf-8", errors="ignore") as file:
        buffer = []
        current_size = 0

        for line in file:
            buffer.append(line)
            current_size += len(line)

            if current_size >= chunk_size:
                yield "".join(buffer)
                buffer = []
                current_size = 0

        if buffer:
            yield "".join(buffer)


def analyze_large_text_file(file_path: Path, chunk_size: int = 5000) -> dict[str, float | int | str]:
    """Analyze sentiment in chunks and return overall results."""
    total_weight = 0
    weighted_polarity = 0.0
    weighted_subjectivity = 0.0
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    chunk_count = 0

    for chunk in iter_text_chunks(file_path, chunk_size=chunk_size):
        clean_chunk = chunk.strip()
        if not clean_chunk:
            continue

        blob = TextBlob(clean_chunk)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        weight = len(clean_chunk)

        weighted_polarity += polarity * weight
        weighted_subjectivity += subjectivity * weight
        total_weight += weight
        chunk_count += 1
        sentiment_counts[classify_sentiment(polarity)] += 1

    if total_weight == 0:
        raise ValueError("The input file does not contain analyzable text.")

    overall_polarity = weighted_polarity / total_weight
    overall_subjectivity = weighted_subjectivity / total_weight

    return {
        "chunks_analyzed": chunk_count,
        "positive_chunks": sentiment_counts["positive"],
        "neutral_chunks": sentiment_counts["neutral"],
        "negative_chunks": sentiment_counts["negative"],
        "polarity": overall_polarity,
        "subjectivity": overall_subjectivity,
        "sentiment": classify_sentiment(overall_polarity),
    }


def save_sentiment_plot(results: dict[str, float | int | str], output_path: Path) -> None:
    """Save a Matplotlib bar chart for the sentiment results."""
    labels = ["positive", "neutral", "negative"]
    values = [
        int(results["positive_chunks"]),
        int(results["neutral_chunks"]),
        int(results["negative_chunks"]),
    ]
    colors = ["#2e7d32", "#607d8b", "#c62828"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.title("TextBlob Sentiment Analysis")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of chunks")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
        )

    summary = (
        f"Overall sentiment: {results['sentiment']}\n"
        f"Polarity: {float(results['polarity']):.4f}"
    )
    plt.figtext(0.5, 0.01, summary, ha="center", fontsize=10)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="TextBlob sentiment analyzer for a large text file")
    parser.add_argument("--file", required=True, help="Path to the text file")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Characters per analysis chunk")
    parser.add_argument(
        "--output",
        default="assignments/sl_assignment_2026/sentiment_analysis.png",
        help="Path for the Matplotlib output chart",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    results = analyze_large_text_file(file_path, chunk_size=args.chunk_size)

    print("TextBlob Sentiment Analysis")
    print("---------------------------")
    print(f"File: {file_path}")
    print(f"Chunks analyzed: {results['chunks_analyzed']}")
    print(f"Positive chunks: {results['positive_chunks']}")
    print(f"Neutral chunks: {results['neutral_chunks']}")
    print(f"Negative chunks: {results['negative_chunks']}")
    print(f"Overall polarity: {results['polarity']:.4f}")
    print(f"Overall subjectivity: {results['subjectivity']:.4f}")
    print(f"Overall sentiment: {results['sentiment']}")

    output_path = Path(args.output)
    save_sentiment_plot(results, output_path)
    print(f"Saved Matplotlib chart to: {output_path}")


if __name__ == "__main__":
    main()
