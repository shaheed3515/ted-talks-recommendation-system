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
from scipy import sparse
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

    def build_user_item_matrix(self, n_users: int = 1000, seed: int = 42) -> sparse.csr_matrix:
        """Constructs a sparse user-item interaction matrix for collaborative filtering."""
        if self.df is None:
            raise ValueError("Model not fitted. Call .fit() first.")
        np.random.seed(seed)
        n_talks = len(self.df)
        n_interactions = n_users * 18
        user_ids = np.random.randint(0, n_users, size=n_interactions)
        talk_ids = np.random.randint(0, n_talks, size=n_interactions)
        ratings = np.random.choice([1, 2, 3, 4, 5], size=n_interactions, p=[0.05, 0.10, 0.25, 0.35, 0.25])
        matrix = sparse.csr_matrix((ratings, (user_ids, talk_ids)), shape=(n_users, n_talks))
        return matrix

    def save_models(self, model_dir: str = "models"):
        """Saves trained vectorizer, similarity matrix, sparse representations, and lookup tables."""
        os.makedirs(model_dir, exist_ok=True)
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        meta_path = os.path.join(model_dir, "ted_metadata.pkl")
        sim_path = os.path.join(model_dir, "similarity_matrix.pkl")
        tfidf_mat_path = os.path.join(model_dir, "tfidf_matrix.npz")
        user_item_path = os.path.join(model_dir, "user_item_matrix.npz")

        print(f"Saving vectorizer to {vec_path}...")
        with open(vec_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"Saving metadata to {meta_path}...")
        with open(meta_path, "wb") as f:
            pickle.dump(self.df[["title", "main_speaker", "views", "duration_minutes", "tags_string", "url", "description"]], f)

        print(f"Saving similarity matrix to {sim_path}...")
        with open(sim_path, "wb") as f:
            pickle.dump(self.similarity_matrix, f)

        print(f"Saving sparse TF-IDF matrix to {tfidf_mat_path}...")
        sparse.save_npz(tfidf_mat_path, self.tfidf_matrix)

        print(f"Generating and saving sparse user-item interaction matrix to {user_item_path}...")
        user_item_matrix = self.build_user_item_matrix()
        sparse.save_npz(user_item_path, user_item_matrix)

        sparsity = 1.0 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
        print(f"User-Item Matrix Sparsity: {sparsity * 100:.2f}%")
        print("All models and matrices successfully saved!")


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
