# SpringBoard Internship Program: Milestone 1 Progress Report
## Project: Personalized TED Talks Recommendation System Using Machine Learning
**Date:** October 2026  
**Status:** Milestone 1 (Weeks 1–2: Modules 1 & 2) Completed

---

## 1. Executive Summary
This document provides a comprehensive report of the progress made during **Milestone 1 (Weeks 1–2)**. It details the data collection, data quality inspection, handling of missing values (`NaN`), text cleaning, data preprocessing, and exploratory data analysis (EDA). The goal is to establish a high-quality, normalized dataset for downstream feature extraction, vectorization, and recommendation modeling.

---

## 2. Dataset Collection & Overview (Module 1)

### 2.1 Raw Files
The dataset was acquired from TED.com archives via Kaggle and stored under `data/raw/`:

| File Name | File Size | Primary Purpose | Record Count | Number of Columns |
| :--- | :--- | :--- | :--- | :--- |
| **`ted_main.csv`** | 7.3 MB | Talk metadata, speakers, tags, views, ratings | 2,550 rows | 17 columns |
| **`transcripts.csv`** | 28.0 MB | Word-for-word spoken transcripts | 2,467 rows | 2 columns |

### 2.2 Merged Dataset Overview
The two files were merged using the unique talk `url` as the primary key:
- **Merged Shape:** 2,553 rows $\times$ 18 columns (prior to duplicate dropping).
- **Final Clean Unique Talks:** 2,550 talks.

### 2.3 Raw Schema Details

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `url` | string | Unique link to the TED Talk page (Primary Key) |
| `title` | string | Official title of the talk |
| `main_speaker` | string | Primary speaker presenting the talk |
| `speaker_occupation` | string | Profession or title of the main speaker |
| `description` | string | Summary synopsis of the talk |
| `duration` | integer | Talk duration in seconds |
| `event` | string | TED event or conference where delivered |
| `film_date` | integer | Unix timestamp of when the talk was recorded |
| `published_date` | integer | Unix timestamp of when published on TED.com |
| `languages` | integer | Number of language subtitle translations available |
| `comments` | integer | Number of user comments on the talk |
| `views` | integer | Total view count |
| `num_speaker` | integer | Number of speakers in the talk |
| `tags` | string | Stringified Python list of topic tags |
| `ratings` | string | Stringified dictionary of viewer sentiment votes |
| `related_talks` | string | List of recommended related talk metadata |
| `transcript` | string | Complete spoken speech transcript |

---

## 3. Data Quality Assessment (The 4 Core Dimensions)

To prevent poor recommendation quality (**Garbage In – Garbage Out**), an exhaustive data quality audit was conducted across the 4 fundamental Data Quality Dimensions:

### 3.1 Dimension 1: Completeness
- **Definition:** Ensures all necessary data attributes are populated without detrimental omissions.
- **Analysis:** Out of 18 merged columns, 16 columns achieved 100% completeness.
  - `transcript`: 86 nulls (3.37% missing).
  - `speaker_occupation`: 6 nulls (0.24% missing).
- **Remediation:** Missing transcripts were imputed using talk descriptions; missing occupations were assigned `'Unknown'`.
- **Chart Deliverable:** `docs/figures/01_completeness_chart.png`

| Column Name | Missing Count (`NaN`) | Missing Percentage | Imputation / Resolution Strategy |
| :--- | :--- | :--- | :--- |
| `transcript` | **86** | 3.37% | **Content Imputation:** Substituted with `description` so no talk is discarded. |
| `speaker_occupation` | **6** | 0.24% | **Default Value Imputation:** Filled with default value `'Unknown'`. |
| `description` | **0** | 0.00% | Verified complete; fallback to empty string if null. |
| `title` | **0** | 0.00% | No missing titles; 100% complete. |
| `main_speaker` | **0** | 0.00% | No missing speaker names; 100% complete. |
| `tags` | **0** | 0.00% | No missing tag fields. |
| `duration` | **0** | 0.00% | No zero or negative durations detected. |
| `views` | **0** | 0.00% | Complete integer counts. |

### 3.2 Dimension 2: Validity
- **Definition:** Confirms that data conforms to standard formats, expected value domains, and data types.
- **Analysis:**
  - Talk `duration` was checked for impossible negative, zero, or extreme values. 99.8% of talks fall in the valid range of 3 to 60 minutes (average 13.8 mins).
  - Timestamps (`film_date`, `published_date`) were validated to confirm they map to realistic calendar dates between 1972 and 2017.
- **Chart Deliverable:** `docs/figures/02_validity_durations.png`

### 3.3 Dimension 3: Accuracy
- **Definition:** Validates that the recorded data faithfully reflects real-world events without anomaly or transcription corruption.
- **Analysis:** Evaluated speech pacing by computing **Words Per Minute (WPM)**:
  $$\text{WPM} = \frac{\text{Transcript Word Count}}{\text{Duration (Minutes)}}$$
  The linear regression between spoken words and duration confirms an average pacing of **148 Words Per Minute**, perfectly aligning with natural human public speaking benchmarks (130–160 WPM).
- **Chart Deliverable:** `docs/figures/03_accuracy_speech_rate.png`

### 3.4 Dimension 4: Consistency
- **Definition:** Ensures cross-table integrity, absence of conflicting duplicate records, and standardized categorical values.
- **Analysis:**
  - **Cross-Table Foreign Key Integrity:** The `url` field successfully joined `ted_main.csv` with `transcripts.csv`.
  - **Uniqueness Check:** Detected and eliminated 3 duplicate URLs (`drop_duplicates(subset=['url'])`).
  - **Categorical Standardization:** Raw unparsed string lists (`"['technology', 'science']"`) were parsed and standardized into lowercased token sets.
- **Chart Deliverable:** `docs/figures/04_consistency_tag_distribution.png`

---

## 4. Data Preprocessing & Cleaning Procedures (Module 2)

All cleaning steps are implemented in `src/preprocess.py` and saved to `data/processed/ted_talks_cleaned.csv`:

### 4.1 Stage Cue & Noise Removal
TED transcripts contain frequent audience interaction markers that introduce false text similarity:
- Cues like `(Laughter)`, `(Applause)`, and `[Music]` were stripped using regex: `re.sub(r'\([A-Za-z\s]+\)', ' ', text)`.
- Standardized text to **lowercase**.
- Removed non-alphanumeric special characters and normalized erratic spacing.

### 4.2 Structured Parsing of Stringified Tags
- Raw tags were stored as string literals: e.g. `"['creativity', 'education', 'children']"`.
- Parsed into native Python lists using `ast.literal_eval`.
- Created a flattened keyword string (`tags_string`) for each talk: `creativity education children`.

### 4.3 Feature Engineering & Transformations
1. **Duration in Minutes:** Converted `duration` from seconds to minutes:
   $$\text{duration\_minutes} = \frac{\text{duration}}{60.0}$$
2. **Transcript Word Count:** Computed `transcript_word_count` to evaluate speech length.
3. **Date Conversion:** Converted `film_date` and `published_date` Unix timestamps to human-readable datetime formats.
4. **Content DNA Construction (`combined_content`):**
   A unified text field was synthesized to represent the core conceptual identity of each talk:
   $$\text{combined\_content} = \text{Title} + \text{Main Speaker} + \text{Tags} + \text{Clean Description} + \text{Transcript (Top 500 Words)}$$

---

## 5. Exploratory Data Analysis (EDA) Findings

Visualizations and statistical summaries generated in `notebooks/01_data_preprocessing.ipynb`:

### 5.1 Dataset Scale
- **Total Processed Talks:** 2,550
- **Total Processed File Size:** ~69 MB (`data/processed/ted_talks_cleaned.csv`)
- **Average Talk Duration:** 13.8 minutes (Median: 12.1 minutes, Range: 2.2 to 87.6 minutes)
- **Average Word Count:** 2,044 words per transcript

### 5.2 Top 10 Most Frequent Topic Tags
The dataset represents a diverse topic spectrum:
1. `technology`: 727 talks
2. `science`: 567 talks
3. `global issues`: 501 talks
4. `culture`: 486 talks
5. `tedx`: 450 talks
6. `design`: 418 talks
7. `business`: 348 talks
8. `entertainment`: 299 talks
9. `health`: 236 talks
10. `innovation`: 229 talks

### 5.3 Top 5 Most Viewed Talks
1. **"Do schools kill creativity?"** by *Ken Robinson* — 47,227,110 views
2. **"Your body language may shape who you are"** by *Amy Cuddy* — 43,155,405 views
3. **"How great leaders inspire action"** by *Simon Sinek* — 34,309,432 views
4. **"The power of vulnerability"** by *Brené Brown* — 31,168,150 views
5. **"10 things you didn't know about orgasm"** by *Mary Roach* — 22,270,883 views

---

## 6. Milestone 1 Verification Checklist

| Evaluation Criterion from Syllabus | Status | Evidence / Location |
| :--- | :---: | :--- |
| **Approval of collected TED Talks dataset, ensuring richness & diversity** | **PASSED** | 2,550 talks in `data/raw/ted_main.csv` covering 400+ distinct tags. |
| **Approval of exploratory analysis (talk length distributions, topic tags)** | **PASSED** | Documented & plotted in `notebooks/01_data_preprocessing.ipynb`. |
| **Approval of preprocessing pipeline for cleaning transcripts & metadata** | **PASSED** | Modular code in `src/preprocess.py`; cleaned CSV in `data/processed/`. |

---

## 7. Next Steps: Milestone 2 Roadmap (Weeks 3–4)
- **Module 3: Feature Engineering and Vectorization**
  - Implement TF-IDF vectorization and word embeddings (Word2Vec / Sentence-BERT).
  - Compute Cosine Similarity matrices over `combined_content`.
- **Module 4: Recommendation Engine**
  - Construct the content-based recommendation function: Given a Talk Title or Tag, return Top-$N$ similar talks.
  - Implement collaborative filtering matrix factorization.
