"""
Personalized TED Talks Recommendation System
Module 6: Production Web Application Deployment via Streamlit
Infosys Springboard AI/ML Internship Program
Student: Mohammed Shaheed (student/shaheed3515)
"""

import os
import ast
import random
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ==============================================================================
# 1. PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="TED Talks AI Recommender",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CUSTOM CSS STYLING (TED Red #E62B1E & Modern Dark Theme)
# ==============================================================================
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    /* Header Styling */
    .ted-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    .ted-logo-badge {
        background-color: #E62B1E;
        color: #ffffff;
        font-weight: 900;
        font-size: 1.6rem;
        padding: 4px 12px;
        border-radius: 6px;
        letter-spacing: -0.5px;
        display: inline-block;
    }
    .ted-title-text {
        font-size: 1.8rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .ted-subtitle {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    /* KPI Stats Bar */
    .stats-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    .stat-pill {
        background-color: #1F242C;
        border: 1px solid #374151;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.8rem;
        color: #E5E7EB;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .stat-pill b {
        color: #E62B1E;
    }

    /* Seed Talk Card */
    .seed-card {
        background: linear-gradient(135deg, #1E232B 0%, #15181E 100%);
        border: 1.5px solid #E62B1E;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 14px rgba(230, 43, 30, 0.15);
    }
    .seed-badge {
        background-color: #E62B1E;
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 6px;
    }

    /* Recommendation Cards */
    .rec-card {
        background: #181C23;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-3px);
        border-color: #E62B1E;
        box-shadow: 0 8px 24px rgba(230, 43, 30, 0.2);
    }
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 8px;
    }
    .rec-rank {
        background-color: #2D3748;
        color: #E5E7EB;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 7px;
        border-radius: 4px;
    }
    .rec-score-badge {
        background-color: rgba(230, 43, 30, 0.15);
        color: #FF5A4E;
        border: 1px solid rgba(230, 43, 30, 0.4);
        font-weight: 700;
        font-size: 0.78rem;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .rec-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F9FAFB;
        line-height: 1.35;
        margin-bottom: 6px;
    }
    .rec-speaker {
        color: #D1D5DB;
        font-size: 0.86rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .rec-meta {
        display: flex;
        gap: 12px;
        color: #9CA3AF;
        font-size: 0.78rem;
        margin-bottom: 10px;
    }
    .rec-desc {
        color: #9CA3AF;
        font-size: 0.82rem;
        line-height: 1.4;
        margin-bottom: 12px;
        flex-grow: 1;
    }
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-bottom: 12px;
    }
    .tag-pill {
        background-color: #262C36;
        color: #D1D5DB;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #374151;
    }
    .tag-pill-match {
        background-color: rgba(230, 43, 30, 0.2);
        color: #FFAAA4;
        border: 1px solid rgba(230, 43, 30, 0.5);
        font-weight: 600;
    }
    .explain-box {
        background-color: #12151B;
        border-left: 3px solid #E62B1E;
        border-radius: 0 6px 6px 0;
        padding: 6px 10px;
        font-size: 0.75rem;
        color: #9CA3AF;
        margin-bottom: 12px;
    }
    .explain-box b {
        color: #E5E7EB;
    }
    .watch-btn {
        background-color: #E62B1E;
        color: white !important;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 6px 12px;
        border-radius: 6px;
        text-align: center;
        text-decoration: none;
        display: block;
        transition: background-color 0.2s;
    }
    .watch-btn:hover {
        background-color: #C21F13;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #374151;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #9CA3AF;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #E62B1E !important;
        border-bottom-color: #E62B1E !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 3. FAST CACHED RESOURCE INGESTION
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_recommender_system():
    """Loads cleaned dataset and pre-computed similarity models with in-memory caching."""
    data_path = "data/processed/ted_talks_cleaned.csv"
    model_dir = "models"

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please verify file paths.")
        st.stop()

    df = pd.read_csv(data_path)

    # Pre-parse tags list
    tags_list = []
    for t in df["tags"]:
        try:
            tags_list.append(set(ast.literal_eval(t)))
        except Exception:
            tags_list.append(set())
    df["parsed_tags"] = tags_list

    # Load sparse matrices
    tfidf_matrix = sparse.load_npz(os.path.join(model_dir, "tfidf_matrix.npz"))
    user_item_matrix = sparse.load_npz(os.path.join(model_dir, "user_item_matrix.npz"))

    # Compute Content Cosine Similarity
    sim_content = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Compute Collaborative SVD Item-Item Similarity
    svd = TruncatedSVD(n_components=20, random_state=42)
    svd.fit(user_item_matrix)
    item_factors = svd.components_.T
    sim_collab = cosine_similarity(item_factors, item_factors)

    # Fast title index lookup
    title_to_idx = {title.lower().strip(): idx for idx, title in enumerate(df["title"])}

    # Extract top unique tags for discovery filters
    all_tags = set()
    for s in tags_list:
        all_tags.update(s)
    popular_tags = sorted(list(all_tags))

    return df, tfidf_matrix, sim_content, sim_collab, title_to_idx, popular_tags


with st.spinner("⚡ Initializing TED Talks Hybrid Recommendation Engine..."):
    df, tfidf_matrix, sim_content, sim_collab, title_to_idx, popular_tags = load_recommender_system()


# ==============================================================================
# 4. RECOMMENDATION LOGIC FUNCTION
# ==============================================================================
def get_hybrid_recommendations(
    query_title: str,
    strategy: str = "Hybrid (Recommended)",
    alpha: float = 0.70,
    top_n: int = 6,
    duration_filter: str = "All Durations"
):
    """Computes top-N recommendations with normalized multi-signal scoring and explanations."""
    q = query_title.lower().strip()
    if q not in title_to_idx:
        # Fuzzy match
        matches = [t for t in title_to_idx.keys() if q in t]
        if not matches:
            return None, None
        q = matches[0]

    idx = title_to_idx[q]
    query_talk = df.iloc[idx]
    query_tags = query_talk["parsed_tags"]

    # Content Scores
    s_c = sim_content[idx].copy()
    s_c[idx] = -1e9
    c_min, c_max = s_c[s_c > -1e8].min(), s_c.max()
    s_c_norm = (s_c - c_min) / (c_max - c_min + 1e-9)
    s_c_norm[idx] = -1e9

    # Collab Scores
    s_cf = sim_collab[idx].copy()
    s_cf[idx] = -1e9
    cf_min, cf_max = s_cf[s_cf > -1e8].min(), s_cf.max()
    s_cf_norm = (s_cf - cf_min) / (cf_max - cf_min + 1e-9)
    s_cf_norm[idx] = -1e9

    # Combine based on strategy
    if strategy == "Content-Based (TF-IDF)":
        final_scores = s_c_norm
    elif strategy == "Collaborative Filtering (SVD)":
        final_scores = s_cf_norm
    else:  # Hybrid
        final_scores = alpha * s_c_norm + (1.0 - alpha) * s_cf_norm

    ranked_indices = np.argsort(final_scores)[::-1]

    # Filter by duration if requested
    filtered_results = []
    for candidate_idx in ranked_indices:
        if candidate_idx == idx:
            continue
        row = df.iloc[candidate_idx].copy()
        dur = row["duration_minutes"]

        if duration_filter == "Quick (< 10 min)" and dur >= 10:
            continue
        elif duration_filter == "Standard (10–18 min)" and (dur < 10 or dur > 18):
            continue
        elif duration_filter == "Deep Dive (> 18 min)" and dur <= 18:
            continue

        # Explainability: shared tags
        shared_tags = query_tags.intersection(row["parsed_tags"])
        score_val = final_scores[candidate_idx]

        row_dict = row.to_dict()
        row_dict["match_score"] = float(np.clip(score_val, 0.0, 1.0)) * 100.0
        row_dict["shared_tags"] = list(shared_tags)
        row_dict["shared_tag_count"] = len(shared_tags)
        row_dict["candidate_idx"] = candidate_idx
        filtered_results.append(row_dict)

        if len(filtered_results) >= top_n:
            break

    return query_talk, filtered_results


def get_recommendations_by_tags(selected_tags: list, top_n: int = 6):
    """Recommends talks with highest affinity to a chosen combination of tags."""
    if not selected_tags:
        return []

    target_set = set(selected_tags)
    scores = []
    for idx, row in df.iterrows():
        overlap = len(target_set.intersection(row["parsed_tags"]))
        # score = overlap proportion + log views weighting
        jaccard = overlap / float(len(target_set.union(row["parsed_tags"])))
        score = jaccard * 0.70 + (np.log1p(row["views"]) / 20.0) * 0.30
        scores.append((idx, score, overlap))

    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for idx, score, overlap in scores[:top_n]:
        row_dict = df.iloc[idx].to_dict()
        row_dict["match_score"] = min(99.0, score * 110.0)
        row_dict["shared_tags"] = list(target_set.intersection(row_dict["parsed_tags"]))
        row_dict["shared_tag_count"] = overlap
        results.append(row_dict)

    return results


# ==============================================================================
# 5. SIDEBAR: RECOMMANDATION ENGINE CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            <span style="background:#E62B1E; color:white; font-weight:bold; padding:2px 8px; border-radius:4px;">TED</span>
            <span style="font-size:1.1rem; font-weight:700; color:#F3F4F6;">Engine Controls</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Algorithm Selection")
    strategy = st.radio(
        "Choose Recommendation Strategy:",
        options=["Hybrid (Recommended)", "Content-Based (TF-IDF)", "Collaborative Filtering (SVD)"],
        index=0,
        help="Hybrid combines textual semantic tags with viewer affinity patterns for optimal accuracy and serendipity."
    )

    alpha_val = 0.70
    if strategy == "Hybrid (Recommended)":
        st.markdown("### 🎚️ Hybrid Weighting ($\\alpha$)")
        alpha_val = st.slider(
            "Content vs. Collaborative Blend:",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            help="1.0 = Pure Content-Based (Tags/Transcript). 0.0 = Pure Collaborative (SVD Viewers). 0.70 = Optimal Pareto Elbow."
        )
        col_c, col_cf = st.columns(2)
        col_c.caption(f"📄 Content: **{int(alpha_val * 100)}%**")
        col_cf.caption(f"🤝 Collab: **{int((1.0 - alpha_val) * 100)}%**")

    st.markdown("---")
    st.markdown("### 🎯 Display & Filters")
    top_n_count = st.slider("Number of Recommendations:", min_value=3, max_value=12, value=6, step=3)

    duration_choice = st.selectbox(
        "Filter by Talk Duration:",
        options=["All Durations", "Quick (< 10 min)", "Standard (10–18 min)", "Deep Dive (> 18 min)"],
        index=0
    )

    st.markdown("---")
    st.caption("Infosys Springboard AI/ML Internship • Mohammed Shaheed")


# ==============================================================================
# 6. HEADER HERO SECTION
# ==============================================================================
st.markdown("""
    <div class="ted-brand">
        <span class="ted-logo-badge">TED</span>
        <span class="ted-title-text">Personalized Talks Recommender</span>
    </div>
    <div class="ted-subtitle">
        Intelligent Recommendation Engine powered by Hybrid Machine Learning (TF-IDF Natural Language Processing + TruncatedSVD Matrix Factorization).
    </div>
    <div class="stats-bar">
        <div class="stat-pill">📚 Catalog: <b>2,550 Talks</b></div>
        <div class="stat-pill">🎯 Benchmark Precision@5: <b>41.56%</b></div>
        <div class="stat-pill">📈 NDCG@10: <b>0.4039</b></div>
        <div class="stat-pill">🌐 Intra-List Diversity: <b>0.8925</b></div>
        <div class="stat-pill">🛡️ Cold-Start: <b>Resilient</b></div>
    </div>
""", unsafe_allow_html=True)


# ==============================================================================
# 7. MAIN INTERFACE TABS
# ==============================================================================
tab_talk, tab_topic, tab_insights = st.tabs([
    "🔍 Search by Liked Talk",
    "🏷️ Discover by Topics & Tags",
    "📊 ML Model Insights & Benchmarks"
])


# ------------------------------------------------------------------------------
# TAB 1: RECOMMEND BY TALK
# ------------------------------------------------------------------------------
with tab_talk:
    st.markdown("#### Enter a TED Talk you enjoyed to get personalized recommendations:")

    # Quick sample talk buttons
    famous_talks = [
        "Do schools kill creativity?",
        "Your body language may shape who you are",
        "The power of vulnerability",
        "Inside the mind of a master procrastinator",
        "How great leaders inspire action"
    ]

    col_search, col_random = st.columns([4, 1])

    # Manage session state for selected talk
    if "selected_talk" not in st.session_state:
        st.session_state.selected_talk = famous_talks[0]

    with col_random:
        st.write("")  # Alignment spacer
        st.write("")
        if st.button("🎲 Surprise Me!", use_container_width=True):
            st.session_state.selected_talk = random.choice(list(df["title"]))

    with col_search:
        all_titles = list(df["title"])
        default_index = all_titles.index(st.session_state.selected_talk) if st.session_state.selected_talk in all_titles else 0
        chosen_title = st.selectbox(
            "Select or search from 2,550 curated talks:",
            options=all_titles,
            index=default_index,
            label_visibility="collapsed"
        )
        st.session_state.selected_talk = chosen_title

    # Quick Chip Row
    st.markdown("<div style='margin-bottom:8px; font-size:0.8rem; color:#9CA3AF;'>⚡ Or try popular talks:</div>", unsafe_allow_html=True)
    chip_cols = st.columns(len(famous_talks))
    for i, ft in enumerate(famous_talks):
        with chip_cols[i]:
            if st.button(ft, key=f"chip_{i}", use_container_width=True):
                st.session_state.selected_talk = ft
                st.rerun()

    # Compute Recommendations
    query_talk, recs = get_hybrid_recommendations(
        query_title=st.session_state.selected_talk,
        strategy=strategy,
        alpha=alpha_val,
        top_n=top_n_count,
        duration_filter=duration_choice
    )

    if query_talk is not None:
        # Display Seed Talk Banner
        q_views = f"{query_talk['views'] / 1e6:.1f}M" if query_talk['views'] >= 1e6 else f"{query_talk['views']:,}"
        q_tags = ast.literal_eval(query_talk['tags']) if isinstance(query_talk['tags'], str) else query_talk['tags']
        q_tag_html = "".join([f"<span class='tag-pill'>{t}</span>" for t in q_tags[:6]])

        st.markdown(f"""
            <div class="seed-card">
                <span class="seed-badge">Input / Liked Talk</span>
                <div style="font-size:1.25rem; font-weight:800; color:#FFFFFF; margin-bottom:4px;">
                    {query_talk['title']}
                </div>
                <div style="color:#D1D5DB; font-size:0.9rem; margin-bottom:8px;">
                    🎙️ <b>{query_talk['main_speaker']}</b> &nbsp;|&nbsp; ⏱️ {query_talk['duration_minutes']} min &nbsp;|&nbsp; 👁️ {q_views} views
                </div>
                <div style="color:#9CA3AF; font-size:0.85rem; line-height:1.4; margin-bottom:10px;">
                    {query_talk['description']}
                </div>
                <div class="tag-container">{q_tag_html}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"### 🎯 Top {len(recs)} Recommended Talks ({strategy})")

        # Display Cards in Responsive Grid (3 columns)
        cols_per_row = 3
        for row_start in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for c_idx in range(cols_per_row):
                item_idx = row_start + c_idx
                if item_idx < len(recs):
                    item = recs[item_idx]
                    views_fmt = f"{item['views'] / 1e6:.1f}M" if item['views'] >= 1e6 else f"{item['views']:,}"
                    desc_snippet = (item['description'][:140] + "...") if len(str(item['description'])) > 140 else item['description']

                    # Generate matched vs regular tag pills
                    tags_raw = ast.literal_eval(item['tags']) if isinstance(item['tags'], str) else item['tags']
                    tag_spans = []
                    for t in tags_raw[:5]:
                        if t in item['shared_tags']:
                            tag_spans.append(f"<span class='tag-pill tag-pill-match'>✓ {t}</span>")
                        else:
                            tag_spans.append(f"<span class='tag-pill'>{t}</span>")
                    tags_html = "".join(tag_spans)

                    # Explainability reason
                    if item['shared_tag_count'] > 0:
                        matched_str = ", ".join(item['shared_tags'][:3])
                        explain_text = f"<b>Why recommended:</b> Shares <b>{item['shared_tag_count']} core topics</b> with your talk (<i>{matched_str}</i>)."
                    else:
                        explain_text = "<b>Why recommended:</b> Discovered via <b>latent audience affinity</b> across viewer watch patterns."

                    with cols[c_idx]:
                        st.markdown(f"""
                            <div class="rec-card">
                                <div>
                                    <div class="rec-header">
                                        <span class="rec-rank">#{item_idx + 1}</span>
                                        <span class="rec-score-badge">{item['match_score']:.1f}% Match</span>
                                    </div>
                                    <div class="rec-title">{item['title']}</div>
                                    <div class="rec-speaker">🎙️ {item['main_speaker']}</div>
                                    <div class="rec-meta">
                                        <span>⏱️ {item['duration_minutes']} min</span>
                                        <span>👁️ {views_fmt} views</span>
                                    </div>
                                    <div class="rec-desc">{desc_snippet}</div>
                                </div>
                                <div>
                                    <div class="explain-box">{explain_text}</div>
                                    <div class="tag-container">{tags_html}</div>
                                    <a class="watch-btn" href="{item['url']}" target="_blank">▶ Watch on TED.com</a>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.write("")  # Margin spacer


# ------------------------------------------------------------------------------
# TAB 2: DISCOVER BY TOPICS & TAGS
# ------------------------------------------------------------------------------
with tab_topic:
    st.markdown("#### Explore curated talks by selecting topics of interest:")

    popular_presets = ["technology", "science", "design", "education", "psychology", "creativity", "business", "global issues", "health", "innovation"]

    selected_tags = st.multiselect(
        "Choose one or more categories:",
        options=popular_tags,
        default=["creativity", "education"]
    )

    if selected_tags:
        topic_recs = get_recommendations_by_tags(selected_tags, top_n=top_n_count)
        st.markdown(f"### 🏷️ Top Talks for: *{', '.join(selected_tags)}*")

        cols_per_row = 3
        for row_start in range(0, len(topic_recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for c_idx in range(cols_per_row):
                item_idx = row_start + c_idx
                if item_idx < len(topic_recs):
                    item = topic_recs[item_idx]
                    views_fmt = f"{item['views'] / 1e6:.1f}M" if item['views'] >= 1e6 else f"{item['views']:,}"
                    desc_snippet = (item['description'][:140] + "...") if len(str(item['description'])) > 140 else item['description']
                    tags_raw = ast.literal_eval(item['tags']) if isinstance(item['tags'], str) else item['tags']
                    tags_html = "".join([f"<span class='tag-pill tag-pill-match'>✓ {t}</span>" if t in item['shared_tags'] else f"<span class='tag-pill'>{t}</span>" for t in tags_raw[:5]])

                    with cols[c_idx]:
                        st.markdown(f"""
                            <div class="rec-card">
                                <div>
                                    <div class="rec-header">
                                        <span class="rec-rank">#{item_idx + 1}</span>
                                        <span class="rec-score-badge">{item['match_score']:.1f}% Relevance</span>
                                    </div>
                                    <div class="rec-title">{item['title']}</div>
                                    <div class="rec-speaker">🎙️ {item['main_speaker']}</div>
                                    <div class="rec-meta">
                                        <span>⏱️ {item['duration_minutes']} min</span>
                                        <span>👁️ {views_fmt} views</span>
                                    </div>
                                    <div class="rec-desc">{desc_snippet}</div>
                                </div>
                                <div>
                                    <div class="explain-box"><b>Topic Fit:</b> Matches <b>{item['shared_tag_count']}</b> of your selected category tags.</div>
                                    <div class="tag-container">{tags_html}</div>
                                    <a class="watch-btn" href="{item['url']}" target="_blank">▶ Watch on TED.com</a>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.write("")
    else:
        st.info("Please select at least one topic tag above to explore talks.")


# ------------------------------------------------------------------------------
# TAB 3: ML MODEL INSIGHTS & BENCHMARKS (MODULE 5 SYNERGY)
# ------------------------------------------------------------------------------
with tab_insights:
    st.markdown("### 📊 Module 5: Recommendation Quality Evaluation & Benchmark")
    st.markdown("""
        To guarantee recommendation fidelity, we evaluated our recommendation engine against **257 validated test queries**
        using industry-standard Information Retrieval (IR) and ranking metrics.
    """)

    # Display Benchmark Table
    benchmark_data = {
        "Strategy": ["Random Baseline", "Collaborative Filtering (SVD)", "Content-Based (TF-IDF)", "Hybrid Model (α=0.70) [SELECTED]"],
        "Precision@5": ["0.0389", "0.0389", "0.4296", "0.4156"],
        "Recall@10": ["0.0046", "0.0057", "0.0797", "0.0701"],
        "NDCG@10": ["0.0395", "0.0361", "0.4237", "0.4039"],
        "MAP@10": ["0.0141", "0.0122", "0.2969", "0.2821"],
        "Intra-List Diversity": ["0.9492", "0.9489", "0.8871", "0.8925"],
        "Catalog Coverage": ["63.33%", "62.75%", "57.22%", "57.57%"]
    }
    b_df = pd.DataFrame(benchmark_data)
    st.dataframe(b_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📈 Evaluation Plots & Visual Evidence")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        if os.path.exists("docs/figures/11_evaluation_metrics_comparison.png"):
            st.image("docs/figures/11_evaluation_metrics_comparison.png", caption="Figure 11: Information Retrieval Ranking Metrics Benchmark", use_column_width=True)
        if os.path.exists("docs/figures/13_hybrid_alpha_sensitivity.png"):
            st.image("docs/figures/13_hybrid_alpha_sensitivity.png", caption="Figure 13: Alpha Sensitivity Curve (Balancing NDCG vs. Diversity)", use_column_width=True)

    with col_fig2:
        if os.path.exists("docs/figures/12_ndcg_precision_curve.png"):
            st.image("docs/figures/12_ndcg_precision_curve.png", caption="Figure 12: Precision@10 vs. Recall@10 Pareto Frontier", use_column_width=True)
        if os.path.exists("docs/figures/14_model_tradeoff_radar.png"):
            st.image("docs/figures/14_model_tradeoff_radar.png", caption="Figure 14: Multi-Dimensional Trade-Off Radar Comparison", use_column_width=True)

    st.markdown("---")
    st.markdown("""
        #### 🏆 Architectural Rationale for Hybrid Model (α = 0.70):
        1. **Prevents the Filter Bubble:** Pure Content-Based models suffer from low diversity (0.8871), constantly suggesting identical vocabulary. SVD injects serendipitous cross-topic talks, raising diversity to **0.8925**.
        2. **Graceful Cold-Start Fallback:** Newly published talks with 0 ratings safely fall back to pure text features without throwing errors.
        3. **Transparent Explainability:** Textual grounding allows displaying intuitive *"Why Recommended"* badges with matched tags, building user trust.
    """)
