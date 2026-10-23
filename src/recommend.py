"""
TED Talks Recommendation Engine - Milestone 2 (Modules 3 & 4)
Implements:
- TF-IDF Feature Engineering & Vectorization
- Cosine Similarity Computation
- Content-Based Recommendation Function
- Model Serialization for Deployment
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TedRecommender:
    def __init__(self, data_path: str = "data/processed/ted_talks_cleaned.csv"):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.indices = None

    def fit(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """Loads cleaned data, computes TF-IDF representations, and builds cosine similarity matrix."""
        if not os.path.exists(self.data_path):
            from preprocess import run_pipeline
            print(f"{self.data_path} not found. Running preprocessing pipeline...")
            self.df = run_pipeline(output_path=self.data_path)
        else:
            print(f"Loading cleaned data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)

        # Fill any missing combined_content fallback
        self.df["combined_content"] = self.df["combined_content"].fillna(self.df["title"])

        print(f"Fitting TF-IDF Vectorizer (max_features={max_features}, ngrams={ngram_range})...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_content"])
        print(f"TF-IDF Matrix Shape: {self.tfidf_matrix.shape} (Talks x Features)")

        print("Computing Cosine Similarity Matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print(f"Similarity Matrix Shape: {self.similarity_matrix.shape}")

        # Mapping of talk title (lowercase) to dataframe index
        self.indices = pd.Series(self.df.index, index=self.df["title"].str.lower().str.strip()).drop_duplicates()
        return self

    def recommend(self, title: str, top_n: int = 5) -> pd.DataFrame:
        """Returns top_n recommended TED talks given a talk title."""
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        query = title.lower().strip()
        if query not in self.indices:
            # Fuzzy fallback matching
            matches = [t for t in self.indices.index if query in t]
            if not matches:
                raise KeyError(f"Talk titled '{title}' not found in dataset.")
            query = matches[0]
            print(f"Matched talk: '{query}'")

        idx = self.indices[query]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        # Sort by similarity score descending (excluding the talk itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        talk_indices = [i[0] for i in sim_scores]
        scores = [round(i[1], 4) for i in sim_scores]

        results = self.df.iloc[talk_indices][["title", "main_speaker", "views", "duration_minutes", "tags_string", "url"]].copy()
        results["similarity_score"] = scores
        return results

    def save_models(self, model_dir: str = "models"):
        """Saves trained vectorizer and lookup tables for fast loading in production / Streamlit."""
        os.makedirs(model_dir, exist_ok=True)
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        meta_path = os.path.join(model_dir, "ted_metadata.pkl")
        sim_path = os.path.join(model_dir, "similarity_matrix.pkl")

        print(f"Saving vectorizer to {vec_path}...")
        with open(vec_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"Saving metadata to {meta_path}...")
        with open(meta_path, "wb") as f:
            pickle.dump(self.df[["title", "main_speaker", "views", "duration_minutes", "tags_string", "url", "description"]], f)

        print(f"Saving similarity matrix to {sim_path}...")
        with open(sim_path, "wb") as f:
            pickle.dump(self.similarity_matrix, f)

        print("Models successfully saved!")


if __name__ == "__main__":
    recommender = TedRecommender()
    recommender.fit()
    recommender.save_models()

    print("\n" + "="*70)
    print("DEMO RECOMMENDATION TEST 1: 'Do schools kill creativity?'")
    print("="*70)
    recs1 = recommender.recommend("Do schools kill creativity?", top_n=5)
    for i, (_, row) in enumerate(recs1.iterrows(), 1):
        print(f"{i}. {row['title']} by {row['main_speaker']}")
        print(f"   Similarity: {row['similarity_score']:.4f} | Views: {row['views']:,} | Tags: {row['tags_string'][:60]}...")

    print("\n" + "="*70)
    print("DEMO RECOMMENDATION TEST 2: 'Your body language may shape who you are'")
    print("="*70)
    recs2 = recommender.recommend("Your body language may shape who you are", top_n=5)
    for i, (_, row) in enumerate(recs2.iterrows(), 1):
        print(f"{i}. {row['title']} by {row['main_speaker']}")
        print(f"   Similarity: {row['similarity_score']:.4f} | Views: {row['views']:,} | Tags: {row['tags_string'][:60]}...")
