"""
TED Talks Recommendation System - Data Loading, Preprocessing & Exploration
Module 1 & 2: Milestone 1
"""

import ast
import os
import re
import string
import pandas as pd
import numpy as np


def load_raw_data(raw_data_dir: str = "data/raw") -> pd.DataFrame:
    """Loads ted_main.csv and transcripts.csv, auto-downloading if missing."""
    import urllib.request

    os.makedirs(raw_data_dir, exist_ok=True)
    main_path = os.path.join(raw_data_dir, "ted_main.csv")
    transcripts_path = os.path.join(raw_data_dir, "transcripts.csv")

    urls = {
        main_path: "https://raw.githubusercontent.com/cpethe/TED_Talks/master/ted_main.csv",
        transcripts_path: "https://raw.githubusercontent.com/cpethe/TED_Talks/master/transcripts.csv",
    }

    for path, url in urls.items():
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)} from source...")
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded {os.path.basename(path)} successfully.")

    print(f"Loading metadata from {main_path}...")
    ted_df = pd.read_csv(main_path)
    print(f"Loading transcripts from {transcripts_path}...")
    transcripts_df = pd.read_csv(transcripts_path)

    print(f"Metadata records: {ted_df.shape[0]} rows, {ted_df.shape[1]} columns")
    print(f"Transcripts records: {transcripts_df.shape[0]} rows, {transcripts_df.shape[1]} columns")

    # Merge on 'url'
    merged_df = pd.merge(ted_df, transcripts_df, on="url", how="left")
    print(f"Merged records: {merged_df.shape[0]} rows")
    return merged_df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Performs initial data quality assessment: shape, missing values, duplicates."""
    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_urls": df["url"].duplicated().sum(),
        "transcripts_available": df["transcript"].notnull().sum(),
        "transcripts_missing": df["transcript"].isnull().sum(),
    }
    return summary


def clean_text(text: str, remove_applause_tags: bool = True) -> str:
    """
    Cleans transcript and description text:
    - Removes TED audience annotations like (Laughter), (Applause)
    - Converts to lowercase
    - Normalizes punctuation and excessive whitespace
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""

    # Remove (Laughter), (Applause), [Music], etc.
    if remove_applause_tags:
        text = re.sub(r"\([A-Za-z\s]+\)", " ", text)
        text = re.sub(r"\[[A-Za-z\s]+\]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Replace newlines/tabs with space
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove non-alphanumeric except standard spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_tags(tags_str: str) -> list:
    """Safely parses stringified list of tags into a list of strings."""
    if not isinstance(tags_str, str) or pd.isna(tags_str):
        return []
    try:
        parsed = ast.literal_eval(tags_str)
        if isinstance(parsed, list):
            return [str(t).strip().lower() for t in parsed]
        return [str(parsed).strip().lower()]
    except Exception:
        # Fallback regex extraction
        cleaned = re.sub(r"[\[\]'\"\s]", "", tags_str)
        return [t.lower() for t in cleaned.split(",") if t]


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive preprocessing pipeline:
    1. Fill missing transcripts with descriptions as fallback.
    2. Parse and clean tags.
    3. Calculate talk duration (minutes), transcript word counts.
    4. Convert unix timestamps to human-readable dates.
    5. Clean titles, descriptions, and transcripts.
    6. Construct 'combined_content' feature for downstream NLP & recommendation.
    """
    df = df.copy()

    # Drop duplicate talk URLs if any
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # Handle missing speaker occupation or description
    df["speaker_occupation"] = df["speaker_occupation"].fillna("Unknown")
    df["description"] = df["description"].fillna("")

    # Convert timestamps
    if "published_date" in df.columns:
        df["published_datetime"] = pd.to_datetime(df["published_date"], unit="s", errors="coerce")
    if "film_date" in df.columns:
        df["film_datetime"] = pd.to_datetime(df["film_date"], unit="s", errors="coerce")

    # Duration in minutes
    if "duration" in df.columns:
        df["duration_minutes"] = (df["duration"] / 60.0).round(2)

    # Parse tags
    df["tags_list"] = df["tags"].apply(parse_tags)
    df["num_tags"] = df["tags_list"].apply(len)
    df["tags_string"] = df["tags_list"].apply(lambda tags: " ".join(tags))

    # Text cleaning
    print("Cleaning descriptions and transcripts...")
    df["clean_description"] = df["description"].apply(clean_text)
    
    # Fill missing transcripts with description
    df["raw_transcript"] = df["transcript"].fillna("")
    df["clean_transcript"] = df["raw_transcript"].apply(clean_text)

    # Calculate word and character counts
    df["transcript_word_count"] = df["clean_transcript"].apply(lambda x: len(x.split()) if x else 0)
    df["transcript_char_count"] = df["clean_transcript"].apply(len)

    # Combined content for content-based vectorization (Title + Tags + Speaker + Description + Transcript)
    # Give weight to title and tags by repeating them
    df["combined_content"] = (
        df["title"].astype(str) + " " +
        df["main_speaker"].astype(str) + " " +
        df["tags_string"] + " " +
        df["clean_description"] + " " +
        df["clean_transcript"].apply(lambda x: " ".join(x.split()[:500])) # Top 500 words of transcript for balance
    )

    return df


def run_pipeline(
    raw_dir: str = "data/raw",
    output_path: str = "data/processed/ted_talks_cleaned.csv"
) -> pd.DataFrame:
    """Executes the complete Week 1 data collection, inspection, and preprocessing pipeline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load raw data
    merged_df = load_raw_data(raw_dir)

    # Step 2: Quality Inspection
    inspection = inspect_dataset(merged_df)
    print("\n--- Data Quality Summary ---")
    print(f"Total talks: {inspection['total_records']}")
    print(f"Transcripts present: {inspection['transcripts_available']} (Missing: {inspection['transcripts_missing']})")
    print(f"Duplicate talk URLs: {inspection['duplicate_urls']}")

    # Step 3: Preprocess data
    print("\nExecuting preprocessing pipeline...")
    processed_df = preprocess_dataset(merged_df)

    # Step 4: Save cleaned dataset
    print(f"\nSaving preprocessed dataset to {output_path}...")
    processed_df.to_csv(output_path, index=False)
    print(f"Saved successfully! File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

    # Step 5: Summary statistics
    print("\n--- Top 10 Most Frequent Tags ---")
    all_tags = [tag for tags in processed_df["tags_list"] for tag in tags]
    tag_counts = pd.Series(all_tags).value_counts().head(10)
    for tag, count in tag_counts.items():
        print(f"  • {tag}: {count} talks")

    print("\n--- Top 5 Most Viewed Talks ---")
    top_viewed = processed_df.sort_values(by="views", ascending=False)[["title", "main_speaker", "views"]].head(5)
    for _, row in top_viewed.iterrows():
        print(f"  • {row['title']} by {row['main_speaker']} ({row['views']:,} views)")

    print(f"\nAverage Talk Duration: {processed_df['duration_minutes'].mean():.1f} minutes")
    print(f"Average Transcript Word Count: {processed_df['transcript_word_count'].mean():.0f} words")

    return processed_df


if __name__ == "__main__":
    run_pipeline()
