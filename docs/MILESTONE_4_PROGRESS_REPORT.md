# SpringBoard Internship Program: Milestone 4 Progress Report
## Project: Personalized TED Talks Recommendation System Using Machine Learning
**Module:** Module 6 — Deployment via Streamlit App & Final Presentation  
**Timeline:** Weeks 7–8  
**Student:** Shaheed Shaik (`student/shaheed3515`)  
**Status:** Milestone 4 Completed, Deployed & Fully Verified  

---

## 1. Executive Summary

This report documents the implementation, design architecture, and deployment procedures for **Milestone 4 (Weeks 7–8 / Module 6: Deployment via Streamlit App)** of the *Personalized TED Talks Recommendation System*.

Following the completion of data collection (Module 1), preprocessing (Module 2), feature engineering (Module 3), model training (Module 4), and ranking evaluation (Module 5), Module 6 delivers an interactive, high-performance web application deployed via **Streamlit**:
1. **Interactive Web Interface (`app.py`)**: Designed a modern, responsive web application inspired by TED.com and Netflix aesthetics, featuring TED Red accents (`#E62B1E`), sleek dark-mode card grids, and smooth animations.
2. **Seamless Model Integration**: Integrated all pre-trained model artifacts (`tfidf_matrix.npz`, `user_item_matrix.npz`, TruncatedSVD $k=20$, and cosine similarity matrices) using Streamlit's `@st.cache_resource` for sub-second in-memory inference.
3. **Multi-Modal User Input**:
   - **Search by Liked Talk:** Instant autocomplete search over all 2,550 curated talks, a "Surprise Me!" randomizer, and quick-access chips for famous talks.
   - **Discover by Topics & Tags:** Dynamic multi-category filter allowing users to discover talks based on topic intersections (e.g., Technology + Design + Psychology).
   - **Duration Filtering:** Granular filtering options (Quick $<10$ min, Standard $10–18$ min, Deep Dive $>18$ min).
4. **Algorithmic Flexibility & Transparency**:
   - Live strategy selector: **Hybrid (Recommended)**, **Content-Based (TF-IDF)**, or **Collaborative Filtering (SVD)**.
   - Interactive $\alpha$ slider allowing real-time adjustment of Content vs. Collaborative weights.
   - Transparent *"Why Recommended"* explainability badges highlighting shared topics and latent viewer affinities.
5. **Integrated Model Benchmarks**: Embedded an interactive analytics tab displaying the Module 5 benchmark metrics, evaluation curves, and trade-off radar charts.

---

## 2. Application Architecture & Technical Flow

```
┌────────────────────────────────────────────────────────┐
│             Streamlit Web Interface (app.py)           │
│  - Hero Banner & Quick KPI Counters                    │
│  - Autocomplete Search / Topic Multi-select            │
│  - Strategy Radio & Dynamic Alpha Slider               │
└──────────────────────────┬─────────────────────────────┘
                           │ User Input (Title / Tags / Strategy / Alpha)
                           ▼
┌────────────────────────────────────────────────────────┐
│         Cached In-Memory Inference Pipeline            │
│  - @st.cache_resource (Instant Page Reruns)            │
│  - Scipy Sparse CSR Matrices (.npz)                    │
│  - TruncatedSVD Item Factors (2550 x 20)               │
│  - Pairwise Cosine Similarity Matrices (2550 x 2550)   │
└──────────────────────────┬─────────────────────────────┘
                           │ Fast Vector Operations
                           ▼
┌────────────────────────────────────────────────────────┐
│           Hybrid Recommendation Engine                 │
│  S_hybrid = alpha * S_content + (1 - alpha) * S_collab │
│  - Score Normalization (0.0 to 1.0)                    │
│  - Top-N Ranking (Argsort)                             │
│  - Topic Tag Intersection (Explainability)             │
└──────────────────────────┬─────────────────────────────┘
                           │ Ranked Talk Dictionaries
                           ▼
┌────────────────────────────────────────────────────────┐
│             Responsive Card Grid Display               │
│  - Match Score Badge (e.g., 94.2% Match)               │
│  - Talk Title, Speaker, Views, Duration                │
│  - "Why Recommended" Topic Explanation Box             │
│  - Direct Watch Link to TED.com                        │
└────────────────────────────────────────────────────────┘
```

---

## 3. Key Interface Features & User Experience (UX)

### 3.1 Design Aesthetic & Branding
- **Signature Palette:** Official TED Red (`#E62B1E`), charcoal surface (`#181C23`), and crisp typography using Inter font.
- **Card Grid Layout:** Results render across responsive 3-column grids, preventing visual clutter.
- **Match Score Badges:** Every recommendation displays a prominent percentage match badge (e.g., `96.5% Match`).

### 3.2 Dual Discovery Modes
1. **Search by Liked Talk:** Users select any talk from the 2,550-talk catalog. The application displays the "Seed Talk" banner showing its speaker, tags, and summary, followed by top-$N$ ranked similar talks.
2. **Discover by Topics:** Users select any combination of categories (e.g., `creativity`, `education`, `innovation`). The engine scores talks based on Jaccard tag similarity blended with log-view popularity.

### 3.3 Explainable AI (XAI)
Every recommended card includes a dedicated explanation container:
- **Topical Matches:** *"Why recommended: Shares 4 core topics with your talk (Education, Creativity, Schools, Child Development)."*
- **Collaborative Matches:** *"Why recommended: Discovered via latent audience affinity across viewer watch patterns."*

---

## 4. Local Execution & Deployment Guide

### 4.1 Running Locally
To launch the recommendation application on your local machine:

1. **Activate the Virtual Environment:**
   ```bash
   cd ~/Desktop/TedTalks
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the App:**
   Open your browser and navigate to:  
   👉 `http://localhost:8501`

### 4.2 Cloud Deployment (Streamlit Community Cloud)
The application is pre-configured for free cloud deployment:
1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository: `shaheed3515/ted-talks-recommendation-system`.
4. Set Main file path: `app.py`.
5. Click **Deploy!** The cloud platform installs dependencies from `requirements.txt` and provisions a public live URL.

---

## 5. Milestone 4 Verification & Deliverables Summary

| Syllabus Requirement | Implementation | Status |
| :--- | :--- | :---: |
| **Interactive Web Interface** | Full Streamlit application implemented in `app.py`. | **VERIFIED** |
| **Trained Models Integration** | In-memory loading of TF-IDF, SVD, and user-item matrices. | **VERIFIED** |
| **User Input Options** | Search bar with autocomplete, tag multi-select, randomizer. | **VERIFIED** |
| **Top-N Talk Display** | Cards showing title, speaker, tags, views, duration, link. | **VERIFIED** |
| **Recommendation Explanations** | Tag intersection and latent affinity explainability pills. | **VERIFIED** |
| **Responsiveness & UX** | Custom CSS layout, 3-column card grid, TED-themed branding. | **VERIFIED** |
| **Deployment Documentation** | Comprehensive deployment instructions in progress report. | **VERIFIED** |

---
*Report prepared and submitted under the Infosys Springboard Internship Program.*
