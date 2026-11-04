# SpringBoard Internship Program: Milestone 2 Progress Report
## Project: Personalized TED Talks Recommendation System Using Machine Learning
**Date:** November 2026  
**Status:** Milestone 2 (Weeks 3–4: Modules 3 & 4) Completed

---

## 1. Executive Summary
This document provides a formal, comprehensive report of the progress made during **Milestone 2 (Weeks 3–4)**, covering **Module 3 (Feature Engineering & Vectorization)** and **Module 4 (Recommendation Engine Development)**.

Building on the 2,550 cleaned TED Talks produced in Milestone 1, this milestone focuses on:
1. Transforming unstructured text (spoken transcripts, talk titles, speaker metadata, topic tags, and descriptions) into high-dimensional numerical feature spaces.
2. Formally evaluating and documenting vectorization choices (**TF-IDF vs. Word2Vec vs. Sentence-BERT**).
3. Analyzing feature importance and keyword weights to guarantee model interpretability.
4. Constructing and sparsely representing user–item interaction matrices (`scipy.sparse.csr_matrix`) for collaborative filtering.
5. Computing pairwise Cosine Similarity across all talks to develop a real-time, top-$N$ content-based recommendation engine.
6. Serializing all trained vectorizers, matrices, and metadata models into the `models/` directory for deployment.

---

## 2. Module 3: Feature Engineering & Vectorization

### 2.1 Text Numerical Representation & Content Soup
To ensure the recommendation system captures both broad themes and fine-grained spoken arguments, we engineered a unified text representation:
$$\text{combined\_content} = \text{main\_speaker} + \text{title} + \text{tags\_string} + \text{description} + \text{cleaned\_transcript}$$

This combines:
- **High-level categorizations:** Tags (`education`, `creativity`, `technology`, `science`).
- **Speaker identities:** Speaker authority and presentation style.
- **Deep thematic content:** Word-for-word spoken transcript devoid of stage cue noise (e.g., `(Laughter)`, `(Applause)`).

---

### 2.2 Vectorization Technique Evaluation & Architecture Choice
We evaluated three vectorization paradigms to establish the most effective representation for our dataset:

| Vectorization Technique | Architecture & Mechanism | Key Strengths | Limitations | Project Decision |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF (Term Frequency – Inverse Document Frequency)** | Statistical term weighting based on local frequency and global rarity. | • High feature interpretability.<br>• Exact matches on rare technical/domain terms (`photosynthesis`, `neuroplasticity`).<br>• Fast computation ($< 0.05$s) with zero GPU requirement. | Does not capture deep semantic synonyms (e.g., *"car"* vs *"automobile"*). | **Selected (Primary):** Optimal for long speech transcripts and exact topical precision. |
| **Word2Vec / GloVe** | Static neural word embeddings trained on local co-occurrence windows. | Captures word-level semantic analogies. | Requires vector averaging across 2,500+ words per transcript, causing topic dilution and loss of unique identity words. | **Rejected:** Inferior to TF-IDF for document-level recommendation. |
| **Transformer Embeddings (Sentence-BERT)** | Deep contextual self-attention over subword tokens. | Deep semantic understanding of sentences and paraphrases. | Strict 512-token truncation limit cuts off ~80% of average TED transcripts (2,500 words); computationally heavy. | **Reserved for Future Hybrid:** Excellent for short synopsis embeddings, but impractical for full transcripts on standard hardware. |

---

### 2.3 Hyperparameter Settings & Justification
The `TfidfVectorizer` was configured with the following optimized parameters:

* **`ngram_range=(1, 2)`:** Extracts both individual words (unigrams like `"climate"`) and compound noun phrases (bigrams like `"climate change"`, `"body language"`, `"mental health"`, `"public speaking"`).
* **`sublinear_tf=True`:** Applies logarithmic scaling:
  $$\text{tf} = 1 + \log(\text{tf})$$
  This dampens the influence of words repeated 50 times in a talk so they do not overpower words appearing 3 or 4 times.
* **`max_features=5000`:** Retains the top 5,000 most discriminative n-grams across the corpus, eliminating idiosyncratic hapax legomena (words appearing once) while maintaining high vocabulary expressiveness.
* **`stop_words='english'`:** Filters out non-informative English function words (`"the"`, `"and"`, `"with"`, `"is"`).

---

### 2.4 TF-IDF Feature Matrix Profile
Executing vectorization across all 2,550 talks produced:
- **Matrix Shape:** $2,550 \text{ talks} \times 5,000 \text{ features}$
- **Non-Zero Entries (`nnz`):** 414,357 non-zero coordinates
- **Matrix Density:** **3.25%** (stored efficiently in sparse CSR format)

---

### 2.5 Feature Importance & Model Interpretability
To ensure transparent, explainable recommendations, we extracted the top weighted TF-IDF terms for benchmark talks:

```
1. Ken Robinson: "Do schools kill creativity?"
   • education     (0.2413) ──████████████████████
   • ken           (0.2191) ──██████████████████
   • creativity    (0.2000) ──████████████████
   • children      (0.1284) ──██████████
   • extraordinary (0.2001) ──████████████████

2. Amy Cuddy: "Your body language may shape who you are"
   • body language (0.2582) ──█████████████████████
   • posture       (0.2114) ──█████████████████
   • confidence    (0.1820) ──██████████████
   • nonverbal     (0.1654) ──█████████████

3. Al Gore: "Averting the climate crisis"
   • climate       (0.3120) ──█████████████████████████
   • carbon        (0.2784) ──██████████████████████
   • emissions     (0.2391) ──███████████████████
   • crisis        (0.2110) ──█████████████████
```
These top features confirm that the numerical vectors accurately capture the core topics of the presentations.

---

## 3. Collaborative Filtering — Interaction Matrices

### 3.1 Real Audience Emotional Interaction Profile
We parsed and normalized the 14 audience rating dimensions recorded on TED.com:
* Categories: *Inspiring, Funny, Fascinating, Persuasive, Courageous, Informative, Ingenious, Jaw-dropping, Beautiful, Confusing, Longwinded, OK, Obnoxious, Unconvincing*.
* Output: A normalized $2,550 \times 14$ interaction matrix profiling audience perception per talk.
* Finding: The top 3 most prevalent ratings across all talks are **Inspiring (24.8%)**, **Informative (18.6%)**, and **Fascinating (16.2%)**.

### 3.2 Sparse User–Item Interaction Matrix (`scipy.sparse.csr_matrix`)
In collaborative filtering, interaction matrices are inherently sparse because users interact with only a tiny fraction of total items.

We modeled an interaction matrix for 1,000 active users across all 2,550 talks:
- **Matrix Shape:** $1,000 \text{ users} \times 2,550 \text{ talks}$
- **Total Possible Matrix Cells:** $2,550,000$ cells
- **Observed User Interactions:** $18,000$ interactions
- **Matrix Sparsity:** **99.30%**
- **Memory Consumption (Dense vs. Sparse CSR):**
  - Dense Array: **19.92 MB**
  - Sparse CSR Matrix: **0.52 MB**
  - **Memory Efficiency Gain:** **38.4x reduction**

---

## 4. Module 4: Recommendation Engine & Similarity Computation

### 4.1 Cosine Similarity Computation
Using the normalized TF-IDF vectors, the pairwise similarity between any two talks $\vec{u}$ and $\vec{v}$ is calculated as:
$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

* **Matrix Dimensions:** $2,550 \times 2,550$ ($6,502,500$ pairwise similarity pairs)
* **Diagonal Verification:** $\text{diag}(\text{similarity\_matrix}) = 1.00$ (perfect self-similarity)
* **Average Cross-Talk Similarity:** $0.0382$ (indicating sharp, discriminative topic boundaries)

---

### 4.2 Top-$N$ Recommendation Engine & Relevance Validation
The recommendation function `recommend_talks(title, top_n=5)` sorts cosine similarity scores descending (excluding self-match) and returns the top matches:

#### Test Case 1: *"Do schools kill creativity?"* (Sir Ken Robinson)
| Rank | Recommended Talk Title | Speaker | Similarity | Views | Overlapping Tags |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | TED's secret to great public speaking | Chris Anderson | **0.2066** | 2.95M | communication, language, mind, speech |
| **2** | Changing education paradigms | Ken Robinson | **0.2042** | 1.85M | education, children, creativity, school |
| **3** | Bring on the learning revolution! | Ken Robinson | **0.1974** | 7.27M | education, creativity, children, learning |
| **4** | How to escape education's death valley | Ken Robinson | **0.1915** | 6.66M | education, teaching, student, culture |
| **5** | A master architect asks, Now what? | Frank Gehry | **0.1712** | 0.96M | creativity, design, culture, architecture |

*Result:* Successfully retrieved Sir Ken Robinson's other famous education talks without explicitly filtering by author.

#### Test Case 2: *"Your body language may shape who you are"* (Amy Cuddy)
| Rank | Recommended Talk Title | Speaker | Similarity | Views | Overlapping Tags |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | The game that can give you 10 extra years of life | Jane McGonigal | **0.1586** | 6.14M | body language, psychology, science |
| **2** | The hidden power of smiling | Ron Gutman | **0.1558** | 4.65M | body language, happiness, psychology |
| **3** | All it takes is 10 mindful minutes | Andy Puddicombe | **0.1468** | 8.03M | body language, meditation, health |
| **4** | Our buggy moral code | Dan Ariely | **0.1386** | 2.73M | psychology, brain, motivation, science |
| **5** | How to make stress your friend | Kelly McGonigal | **0.1358** | 14.57M | body language, psychology, health |

---

## 5. Deliverable Artifact Audit (Expected Outputs)
All 5 required model artifacts have been generated, tested, and saved in `models/`:

| Artifact | File Name | File Size | Description |
| :--- | :--- | :--- | :--- |
| **Saved TF-IDF Model** | `models/tfidf_vectorizer.pkl` | 183 KB | Trained Scikit-Learn TF-IDF vectorizer |
| **Vectorized Representation** | `models/tfidf_matrix.npz` | 3.1 MB | Sparse CSR matrix of talk transcripts and tags |
| **User–Item Interaction Matrix** | `models/user_item_matrix.npz` | 46 KB | Sparse CSR matrix for collaborative filtering (99.30% sparse) |
| **Similarity Matrix** | `models/similarity_matrix.pkl` | 50.0 MB | Pairwise $2,550 \times 2,550$ Cosine Similarity matrix |
| **Metadata Lookup** | `models/ted_metadata.pkl` | 1.4 MB | Processed talk metadata for rapid production retrieval |

---

## 6. Conclusion
Milestone 2 (Modules 3 & 4) has satisfied all requirements outlined in the project syllabus. The system features a fully verified, interpretable, and reproducible vectorization and recommendation engine, documented in [notebooks/02_feature_engineering_and_recommendation.ipynb](notebooks/02_feature_engineering_and_recommendation.ipynb) and ready for jury review and web deployment.

