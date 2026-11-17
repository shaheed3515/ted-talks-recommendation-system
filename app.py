"""
Personalized TED Talks Recommendation System
Module 6: Production Web Application Deployment via Streamlit
Infosys Springboard AI/ML Internship Program
Student: Shaheed Shaik (student/shaheed3515)
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
# 2. CUSTOM CSS STYLING (Clean Light / Warm Cream & TED Red Palette)
# ==============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B;
    }

    /* Main Container */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }

    /* Top Header Section */
    .ted-brand-wrapper {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }
    .ted-logo-badge {
        background: linear-gradient(135deg, #E62B1E 0%, #BA1B10 100%);
        color: #FFFFFF !important;
        font-weight: 900;
        font-size: 1.7rem;
        padding: 5px 14px;
        border-radius: 8px;
        letter-spacing: -0.5px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(230, 43, 30, 0.25);
    }
    .ted-title-text {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -0.6px;
        line-height: 1.2;
    }
    .ted-subtitle {
        color: #64748B;
        font-size: 0.96rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    /* KPI Stats Bar */
    .stats-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 1.6rem;
    }
    .stat-pill {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 9999px;
        padding: 6px 16px;
        font-size: 0.82rem;
        color: #334155;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stat-pill b {
        color: #E62B1E;
        font-weight: 700;
    }

    /* Input Seed Talk Card */
    .seed-card {
        background: linear-gradient(135deg, #FFF8F8 0%, #FFFFFF 100%);
        border: 1.5px solid #FCA5A5;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(230, 43, 30, 0.08);
    }
    .seed-badge {
        background-color: #E62B1E;
        color: #FFFFFF !important;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 3px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .seed-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
        line-height: 1.3;
    }
    .seed-speaker {
        color: #475569;
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .seed-desc {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-bottom: 12px;
    }

    /* Recommendation Cards (Clean White / Light Surface) */
    .rec-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-4px);
        border-color: #E62B1E;
        box-shadow: 0 12px 24px -4px rgba(230, 43, 30, 0.12), 0 4px 8px -2px rgba(0, 0, 0, 0.04);
    }
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .rec-rank {
        background-color: #F1F5F9;
        color: #475569;
        font-weight: 700;
        font-size: 0.76rem;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
    }
    .rec-score-badge {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 9999px;
    }
    .rec-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.35;
        margin-bottom: 6px;
        min-height: 44px;
    }
    .rec-speaker {
        color: #475569;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .rec-meta {
        display: flex;
        gap: 14px;
        color: #64748B;
        font-size: 0.8rem;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F1F5F9;
    }
    .rec-desc {
        color: #475569;
        font-size: 0.83rem;
        line-height: 1.5;
        margin-bottom: 12px;
        min-height: 55px;
    }

    /* Tag Styling */
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 12px;
    }
    .tag-pill {
        background-color: #F8FAFC;
        color: #475569;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 3px 9px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
    }
    .tag-pill-match {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
        font-weight: 600;
    }

    /* Explainability Container */
    .explain-box {
        background-color: #F8FAFC;
        border-left: 3.5px solid #E62B1E;
        border-radius: 0 8px 8px 0;
        padding: 8px 12px;
        font-size: 0.78rem;
        color: #475569;
        margin-bottom: 12px;
        line-height: 1.45;
    }
    .explain-box b {
        color: #0F172A;
    }

    /* Watch Button */
    .watch-btn {
        background: linear-gradient(135deg, #E62B1E 0%, #C21F13 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 0.84rem;
        padding: 8px 14px;
        border-radius: 8px;
        text-align: center;
        text-decoration: none;
        display: block;
        transition: opacity 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 6px rgba(230, 43, 30, 0.2);
    }
    .watch-btn:hover {
        opacity: 0.92;
        box-shadow: 0 4px 12px rgba(230, 43, 30, 0.35);
        color: #FFFFFF !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0 16px 0;
        margin-bottom: 14px;
        border-bottom: 1.5px solid #E2E8F0;
    }
    .sidebar-badge {
        background-color: #E62B1E;
        color: #FFFFFF;
        font-weight: 900;
        font-size: 1.15rem;
        padding: 3px 10px;
        border-radius: 6px;
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0F172A;
    }
    .sidebar-footer {
        padding-top: 20px;
        border-top: 1px solid #E2E8F0;
        font-size: 0.78rem;
        color: #64748B;
        text-align: center;
        margin-top: 20px;
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

    # Filter by duration
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
# 5. SIDEBAR: CLEAN & WELL-ORGANIZED CONTROLS
# ==============================================================================
with st.sidebar:
    # Top Clean Header
    st.markdown("""
        <div class="sidebar-header">
            <span class="sidebar-badge">TED</span>
            <span class="sidebar-title">Recommender Control</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Recommendation Strategy")
    strategy = st.radio(
        "Select Model Type:",
        options=["Hybrid (Recommended)", "Content-Based (TF-IDF)", "Collaborative Filtering (SVD)"],
        index=0,
        label_visibility="collapsed",
        help="Hybrid blends topic tag matching with viewer interaction patterns."
    )

    alpha_val = 0.70
    if strategy == "Hybrid (Recommended)":
        st.markdown("<div style='margin-top:14px; font-weight:700; font-size:0.88rem; color:#0F172A;'>🎚️ Hybrid Weighting (Content vs. Collab)</div>", unsafe_allow_html=True)
        alpha_val = st.slider(
            "Weighting (Alpha):",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            label_visibility="collapsed"
        )
        col_c, col_cf = st.columns(2)
        col_c.markdown(f"<span style='font-size:0.8rem; color:#475569;'>📄 Content: <b>{int(alpha_val * 100)}%</b></span>", unsafe_allow_html=True)
        col_cf.markdown(f"<span style='font-size:0.8rem; color:#475569;'>🤝 Collab: <b>{int((1.0 - alpha_val) * 100)}%</b></span>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:18px 0; border:none; border-top:1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    st.markdown("#### 🎯 Filters & Display")
    top_n_count = st.slider("Number of Recommendations:", min_value=3, max_value=12, value=6, step=3)

    duration_choice = st.selectbox(
        "Talk Duration:",
        options=["All Durations", "Quick (< 10 min)", "Standard (10–18 min)", "Deep Dive (> 18 min)"],
        index=0
    )

    st.markdown("""
        <div class="sidebar-footer">
            <b>Infosys Springboard AI/ML Internship</b><br>
            Developer: <b>Shaheed Shaik</b><br>
            <code>student/shaheed3515</code>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 6. HEADER HERO SECTION (Visible & Prominent in Light Theme)
# ==============================================================================
st.markdown("""
    <div class="ted-brand-wrapper">
        <span class="ted-logo-badge">TED</span>
        <span class="ted-title-text">Personalized Talks Recommender</span>
    </div>
    <div class="ted-subtitle">
        Intelligent Recommendation Engine powered by Hybrid Machine Learning (TF-IDF NLP + TruncatedSVD Matrix Factorization).
    </div>
    <div class="stats-bar">
        <div class="stat-pill">📚 Catalog: <b>2,550 Talks</b></div>
        <div class="stat-pill">🎯 Precision@5: <b>41.56%</b></div>
        <div class="stat-pill">📈 NDCG@10: <b>0.4039</b></div>
        <div class="stat-pill">🌐 Diversity: <b>0.8925</b></div>
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
    st.markdown("<div style='font-size:0.95rem; font-weight:600; color:#334155; margin-bottom:6px;'>Enter a TED Talk you enjoyed to get personalized recommendations:</div>", unsafe_allow_html=True)

    famous_talks = [
        "Do schools kill creativity?",
        "Your body language may shape who you are",
        "The power of vulnerability",
        "Inside the mind of a master procrastinator",
        "How great leaders inspire action"
    ]

    col_search, col_random = st.columns([4.2, 1])

    if "selected_talk" not in st.session_state:
        st.session_state.selected_talk = famous_talks[0]

    with col_random:
        st.write("")
        st.write("")
        if st.button("🎲 Surprise Me!", use_container_width=True):
            st.session_state.selected_talk = random.choice(list(df["title"]))
            st.rerun()

    with col_search:
        all_titles = list(df["title"])
        default_index = all_titles.index(st.session_state.selected_talk) if st.session_state.selected_talk in all_titles else 0
        chosen_title = st.selectbox(
            "Select or search from 2,550 curated talks:",
            options=all_titles,
            index=default_index,
            label_visibility="collapsed"
        )
        if chosen_title != st.session_state.selected_talk:
            st.session_state.selected_talk = chosen_title
            st.rerun()

    # Quick Sample Talk Chips
    st.markdown("<div style='margin-bottom:6px; font-size:0.8rem; color:#64748B;'>⚡ Quick picks (Famous Talks):</div>", unsafe_allow_html=True)
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
                <div class="seed-title">{query_talk['title']}</div>
                <div class="seed-speaker">🎙️ <b>{query_talk['main_speaker']}</b> &nbsp;|&nbsp; ⏱️ {query_talk['duration_minutes']} min &nbsp;|&nbsp; 👁️ {q_views} views</div>
                <div class="seed-desc">{query_talk['description']}</div>
                <div class="tag-container">{q_tag_html}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:1.25rem; font-weight:800; color:#0F172A; margin: 24px 0 16px 0;'>🎯 Top {len(recs)} Recommended Talks ({strategy})</div>", unsafe_allow_html=True)

        # Responsive 3-Column Grid
        cols_per_row = 3
        for row_start in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for c_idx in range(cols_per_row):
                item_idx = row_start + c_idx
                if item_idx < len(recs):
                    item = recs[item_idx]
                    views_fmt = f"{item['views'] / 1e6:.1f}M" if item['views'] >= 1e6 else f"{item['views']:,}"
                    desc_snippet = (item['description'][:140] + "...") if len(str(item['description'])) > 140 else item['description']

                    tags_raw = ast.literal_eval(item['tags']) if isinstance(item['tags'], str) else item['tags']
                    tag_spans = []
                    for t in tags_raw[:5]:
                        if t in item['shared_tags']:
                            tag_spans.append(f"<span class='tag-pill tag-pill-match'>✓ {t}</span>")
                        else:
                            tag_spans.append(f"<span class='tag-pill'>{t}</span>")
                    tags_html = "".join(tag_spans)

                    # Explainability text
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
                        st.write("")


# ------------------------------------------------------------------------------
# TAB 2: DISCOVER BY TOPICS & TAGS
# ------------------------------------------------------------------------------
with tab_topic:
    st.markdown("<div style='font-size:0.95rem; font-weight:600; color:#334155; margin-bottom:8px;'>Explore curated talks by selecting topics of interest:</div>", unsafe_allow_html=True)

    selected_tags = st.multiselect(
        "Choose one or more categories:",
        options=popular_tags,
        default=["creativity", "education"],
        label_visibility="collapsed"
    )

    if selected_tags:
        topic_recs = get_recommendations_by_tags(selected_tags, top_n=top_n_count)
        st.markdown(f"<div style='font-size:1.2rem; font-weight:800; color:#0F172A; margin: 20px 0 14px 0;'>🏷️ Top Talks for: <i>{', '.join(selected_tags)}</i></div>", unsafe_allow_html=True)

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
    st.markdown("<div style='font-size:1.25rem; font-weight:800; color:#0F172A; margin-bottom:6px;'>📊 Module 5: Recommendation Quality Benchmark</div>", unsafe_allow_html=True)
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

    st.markdown("<hr style='margin:20px 0; border:none; border-top:1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    st.markdown("#### 📈 Evaluation Plots & Visual Evidence")

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

    st.markdown("<hr style='margin:20px 0; border:none; border-top:1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    st.markdown("""
        #### 🏆 Architectural Rationale for Hybrid Model (α = 0.70):
        1. **Prevents the Filter Bubble:** Pure Content-Based models suffer from lower diversity (0.8871), constantly suggesting identical vocabulary. SVD injects serendipitous cross-topic talks, raising diversity to **0.8925**.
        2. **Graceful Cold-Start Fallback:** Newly published talks with 0 ratings safely fall back to pure text features without throwing errors.
        3. **Transparent Explainability:** Textual grounding allows displaying intuitive *"Why Recommended"* badges with matched tags, building user trust.
    """)
