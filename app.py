"""
Personalized TED Talks Recommendation System
Module 6: Production Web Application Deployment via Streamlit
Infosys Springboard AI/ML Internship Program
Developer: Shaheed Shaik (student/shaheed3515)
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
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="TED Talks Recommendation Engine",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. BULLETPROOF LIGHT & WARM CREAM THEME (100% Inline Contrast Guaranteed)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0F172A !important;
    }

    /* Force Clean Page Background */
    .stApp {
        background-color: #F8F9FA !important;
    }

    /* 1. HEADER: 100% Transparent, No Black Bar, Click-Through */
    header[data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        height: auto !important;
        pointer-events: none !important; /* Allow clicks to pass through empty header area */
        z-index: 9999 !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
        pointer-events: none !important;
    }

    /* 2. RE-OPEN / EXPAND SIDEBAR BUTTON - Prominently Visible on Top-Left */
    [data-testid="stExpandSidebarButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
        margin-top: 6px !important;
        margin-left: 8px !important;
        padding: 4px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stExpandSidebarButton"]:hover {
        background-color: #FEF2F2 !important;
        border-color: #E62B1E !important;
        transform: scale(1.08) !important;
    }

    [data-testid="stExpandSidebarButton"] button {
        pointer-events: auto !important;
        color: #E62B1E !important;
    }

    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stExpandSidebarButton"] span {
        fill: #E62B1E !important;
        stroke: #E62B1E !important;
        color: #E62B1E !important;
    }

    /* Collapse Sidebar Button (Inside Sidebar Header) */
    [data-testid="stSidebarCollapseButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }

    /* 3. HIDE ONLY DEPLOY BUTTON AND HAMBURGER MENU */
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"],
    [data-testid="stMainMenuButton"],
    [data-testid="stStatusWidget"],
    [data-testid="stToolbarActions"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* 2. MAIN CONTAINER SPACING - TED Logo 100% visible with clean margin */
    .block-container {
        padding-top: 2.0rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1250px !important;
    }

    /* 3. SIDEBAR STYLING & HIGH CONTRAST TEXT */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1.5px solid #E2E8F0 !important;
    }
    [data-testid="stSidebarUserContent"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: calc(100vh - 4.5rem) !important;
        padding-bottom: 1.5rem !important;
    }
    .sidebar-footer-card {
        margin-top: auto !important;
        background: #F8FAFC !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        text-align: center !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    div[role="radiogroup"] label div,
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* 4. UNIVERSAL INPUT & SEARCH BAR STYLING (Guaranteed Visible Typing) */
    input,
    textarea,
    .stTextInput input,
    .stSelectbox input,
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        caret-color: #E62B1E !important; /* Bold red typing cursor */
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        -webkit-text-fill-color: #0F172A !important;
        opacity: 1 !important;
    }

    .stTextInput div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    .stTextInput div[data-baseweb="input"]:focus-within {
        border-color: #E62B1E !important;
        box-shadow: 0 0 0 3px rgba(230, 43, 30, 0.15) !important;
    }

    .stTextInput input::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
        font-weight: 400 !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* 5. CARD HOVER ANIMATIONS */
    .rec-card-box {
        background: #FFFFFF !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 18px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    .rec-card-box:hover {
        transform: translateY(-4px) !important;
        border-color: #E62B1E !important;
        box-shadow: 0 12px 20px -2px rgba(230, 43, 30, 0.12) !important;
    }

    /* 6. TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        border-bottom: 2px solid #E2E8F0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 700 !important;
        color: #64748B !important;
        font-size: 0.95rem !important;
        padding: 8px 18px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #E62B1E !important;
        border-bottom-color: #E62B1E !important;
    }

    /* 7. BUTTON STYLING */
    .watch-button {
        background-color: #E62B1E !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        padding: 8px 14px !important;
        border-radius: 8px !important;
        text-align: center !important;
        text-decoration: none !important;
        display: block !important;
        box-shadow: 0 2px 4px rgba(230, 43, 30, 0.2) !important;
    }
    .watch-button:hover {
        background-color: #BA1B10 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. FAST CACHED RESOURCE INGESTION
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_recommender_system():
    data_path = "data/processed/ted_talks_cleaned.csv"
    model_dir = "models"

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please verify file paths.")
        st.stop()

    df = pd.read_csv(data_path)
    df["url"] = df["url"].astype(str).str.strip()

    # Parse tags
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

    # Extract top unique tags
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

    # Strategy blend
    if strategy == "Content-Based (TF-IDF)":
        final_scores = s_c_norm
    elif strategy == "Collaborative Filtering (SVD)":
        final_scores = s_cf_norm
    else:  # Hybrid
        final_scores = alpha * s_c_norm + (1.0 - alpha) * s_cf_norm

    ranked_indices = np.argsort(final_scores)[::-1]

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


def find_matching_talks(query: str, limit: int = 25):
    """
    Intelligent multi-field search engine across Titles, Speakers, and Tags.
    Allows users to freely type any keyword, speaker name, or topic.
    """
    if not query or not str(query).strip():
        return []
    q = str(query).strip().lower()

    # Priority 1: Exact or substring match in Title
    t_matches = df[df["title"].str.lower().str.contains(q, na=False, regex=False)]

    # Priority 2: Match in Speaker name
    s_matches = df[df["main_speaker"].str.lower().str.contains(q, na=False, regex=False)]

    # Priority 3: Match in Tags
    tag_matches = df[df["tags"].str.lower().str.contains(q, na=False, regex=False)]

    # Merge preserving relevance order, drop duplicate titles
    combined = pd.concat([t_matches, s_matches, tag_matches]).drop_duplicates(subset=["title"])
    return list(combined["title"].head(limit))


# ==============================================================================
# 5. SIDEBAR: ORGANIZED & HIGH-CONTRAST CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; padding-bottom:14px; margin-bottom:14px; border-bottom:1.5px solid #E2E8F0;">
            <span style="background:#E62B1E; color:#FFFFFF; font-weight:900; font-size:1.2rem; padding:4px 10px; border-radius:6px;">TED</span>
            <span style="font-size:1.15rem; font-weight:800; color:#0F172A;">Recommender</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#0F172A; margin-bottom:4px;'>⚙️ Recommendation Strategy</div>", unsafe_allow_html=True)
    strategy = st.radio(
        "Select Model Type:",
        options=["Hybrid (Recommended)", "Content-Based (TF-IDF)", "Collaborative Filtering (SVD)"],
        index=0,
        label_visibility="collapsed"
    )

    alpha_val = 0.70
    if strategy == "Hybrid (Recommended)":
        st.markdown("<div style='margin-top:14px; font-weight:700; font-size:0.85rem; color:#0F172A;'>🎚️ Content vs. Collab Weight</div>", unsafe_allow_html=True)
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

    st.markdown("<hr style='margin:16px 0; border:none; border-top:1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#0F172A; margin-bottom:4px;'>🎯 Display Settings</div>", unsafe_allow_html=True)
    top_n_count = st.slider("Number of Recommendations:", min_value=3, max_value=12, value=6, step=3)

    duration_choice = st.selectbox(
        "Filter by Duration:",
        options=["All Durations", "Quick (< 10 min)", "Standard (10–18 min)", "Deep Dive (> 18 min)"],
        index=0
    )

    st.markdown("""
        <div class="sidebar-footer-card">
            <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:3px;">
                Infosys Springboard
            </div>
            <div style="font-size:0.86rem; font-weight:800; color:#0F172A; margin-bottom:3px;">
                AI/ML Internship Project
            </div>
            <div style="font-size:0.82rem; font-weight:600; color:#475569;">
                Developer: <b style="color:#E62B1E;">Shaheed Shaik</b>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 6. HEADER HERO SECTION (100% Guaranteed High Contrast & Full Logo)
# ==============================================================================
st.markdown("""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">
        <span style="background-color:#E62B1E; color:#FFFFFF; font-weight:900; font-size:1.8rem; padding:4px 14px; border-radius:8px; letter-spacing:-1px; box-shadow: 0 4px 10px rgba(230,43,30,0.25);">TED</span>
        <h1 style="color:#0F172A; font-size:2.0rem; font-weight:850; margin:0; padding:0; letter-spacing:-0.5px; display:inline;">Talks Recommendation Engine</h1>
    </div>
    <div style="color:#475569; font-size:0.98rem; margin-bottom:1.4rem; font-weight:500;">
        Intelligent Personalized Discovery powered by Hybrid Machine Learning (TF-IDF NLP + TruncatedSVD Matrix Factorization).
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:10px; margin-bottom:1.8rem;">
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:9999px; padding:6px 16px; font-size:0.82rem; color:#334155; font-weight:600; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            📚 Catalog: <b style="color:#E62B1E;">2,550 Talks</b>
        </div>
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:9999px; padding:6px 16px; font-size:0.82rem; color:#334155; font-weight:600; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            🎯 Benchmark Precision@5: <b style="color:#E62B1E;">41.56%</b>
        </div>
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:9999px; padding:6px 16px; font-size:0.82rem; color:#334155; font-weight:600; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            📈 NDCG@10: <b style="color:#E62B1E;">0.4039</b>
        </div>
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:9999px; padding:6px 16px; font-size:0.82rem; color:#334155; font-weight:600; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            🌐 Intra-List Diversity: <b style="color:#E62B1E;">0.8925</b>
        </div>
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:9999px; padding:6px 16px; font-size:0.82rem; color:#334155; font-weight:600; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            🛡️ Cold-Start: <b style="color:#2563EB;">Resilient</b>
        </div>
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
    famous_talks = [
        "Do schools kill creativity?",
        "Your body language may shape who you are",
        "The power of vulnerability",
        "Inside the mind of a master procrastinator",
        "How great leaders inspire action"
    ]

    if "selected_talk" not in st.session_state:
        st.session_state.selected_talk = famous_talks[0]

    st.markdown("<div style='font-size:0.96rem; font-weight:700; color:#0F172A; margin-bottom:8px;'>🔍 Search any TED Talk by Title, Speaker, or Topic:</div>", unsafe_allow_html=True)

    col_search, col_random = st.columns([4.2, 1])

    with col_random:
        if st.button("🎲 Surprise Me!", use_container_width=True):
            st.session_state.selected_talk = random.choice(list(df["title"]))
            if "search_bar_input" in st.session_state:
                st.session_state.search_bar_input = ""
            st.rerun()

    with col_search:
        search_val = st.text_input(
            "Search for a talk:",
            placeholder="Type speaker, topic, or title (e.g., Ken Robinson, creativity, happiness, AI)...",
            label_visibility="collapsed",
            key="search_bar_input"
        )

    # Dynamic Search Filter
    query_str = search_val.strip() if search_val else ""
    if query_str:
        matched_talks = find_matching_talks(query_str, limit=20)
        if matched_talks:
            st.markdown(f"<div style='font-size:0.84rem; font-weight:700; color:#475569; margin: 4px 0 2px 0;'>🎯 Found {len(matched_talks)} talks matching <span style='color:#E62B1E;'>\"{query_str}\"</span> — choose one:</div>", unsafe_allow_html=True)
            sel_idx = matched_talks.index(st.session_state.selected_talk) if st.session_state.selected_talk in matched_talks else 0
            chosen_match = st.selectbox(
                "Select from search results:",
                options=matched_talks,
                index=sel_idx,
                label_visibility="collapsed",
                key="search_results_dropdown"
            )
            if chosen_match != st.session_state.selected_talk:
                st.session_state.selected_talk = chosen_match
                st.rerun()
            elif st.session_state.selected_talk not in matched_talks:
                st.session_state.selected_talk = matched_talks[0]
                st.rerun()
        else:
            st.info(f"💡 No talks found directly matching \"{query_str}\". Try searching by speaker name (e.g. 'Ken Robinson'), or select from the popular talks below.")
    else:
        # Quick expandable catalog browser for convenience
        with st.expander("📚 Or browse all 2,550 curated talks alphabetically", expanded=False):
            all_titles = sorted(list(df["title"]))
            default_index = all_titles.index(st.session_state.selected_talk) if st.session_state.selected_talk in all_titles else 0
            chosen_title = st.selectbox(
                "Catalog list:",
                options=all_titles,
                index=default_index,
                label_visibility="collapsed",
                key="catalog_browser_select"
            )
            if chosen_title != st.session_state.selected_talk:
                st.session_state.selected_talk = chosen_title
                st.rerun()

    # Quick Picks
    st.markdown("<div style='margin-top:10px; margin-bottom:6px; font-size:0.8rem; color:#64748B; font-weight:600;'>⚡ Popular benchmark talks:</div>", unsafe_allow_html=True)
    chip_cols = st.columns(len(famous_talks))
    for i, ft in enumerate(famous_talks):
        with chip_cols[i]:
            if st.button(ft, key=f"chip_{i}", use_container_width=True):
                st.session_state.selected_talk = ft
                if "search_bar_input" in st.session_state:
                    st.session_state.search_bar_input = ""
                st.rerun()

    # Recommendations
    query_talk, recs = get_hybrid_recommendations(
        query_title=st.session_state.selected_talk,
        strategy=strategy,
        alpha=alpha_val,
        top_n=top_n_count,
        duration_filter=duration_choice
    )

    if query_talk is not None:
        q_views = f"{query_talk['views'] / 1e6:.1f}M" if query_talk['views'] >= 1e6 else f"{query_talk['views']:,}"
        q_tags = ast.literal_eval(query_talk['tags']) if isinstance(query_talk['tags'], str) else query_talk['tags']
        q_tag_html = "".join([f"<span style='background:#F1F5F9; color:#334155; font-size:0.75rem; font-weight:600; padding:3px 9px; border-radius:6px; border:1px solid #CBD5E1; margin-right:6px;'>{t}</span>" for t in q_tags[:6]])

        # Warm Cream / Light Red Seed Talk Banner
        st.markdown(f"""
            <div style="background:#FFFDFB; border-left:5px solid #E62B1E; border-top:1px solid #E2E8F0; border-right:1px solid #E2E8F0; border-bottom:1px solid #E2E8F0; border-radius:12px; padding:22px 24px; margin-top:16px; margin-bottom:26px; box-shadow:0 4px 16px rgba(0,0,0,0.04);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="background-color:#E62B1E; color:#FFFFFF; font-size:0.72rem; font-weight:800; padding:3px 10px; border-radius:6px; text-transform:uppercase;">INPUT / LIKED TALK</span>
                    <span style="color:#64748B; font-size:0.84rem; font-weight:600;">⏱️ {query_talk['duration_minutes']} min &nbsp;|&nbsp; 👁️ {q_views} views</span>
                </div>
                <div style="font-size:1.35rem; font-weight:850; color:#0F172A; margin-bottom:4px; line-height:1.3;">
                    {query_talk['title']}
                </div>
                <div style="color:#475569; font-size:0.92rem; font-weight:600; margin-bottom:10px;">
                    🎙️ <b>{query_talk['main_speaker']}</b> ({query_talk.get('speaker_occupation', 'TED Speaker')})
                </div>
                <div style="color:#334155; font-size:0.88rem; line-height:1.55; margin-bottom:12px;">
                    {query_talk['description']}
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px;">
                    {q_tag_html}
                </div>
                <div>
                    <a class="watch-button" style="display:inline-block; padding:7px 18px;" href="{query_talk['url']}" target="_blank">▶ Watch this talk on TED.com</a>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:1.3rem; font-weight:850; color:#0F172A; margin: 20px 0 16px 0;'>🎯 Top {len(recs)} Recommended Talks ({strategy})</div>", unsafe_allow_html=True)

        # Responsive 3-Column Card Grid (Clean White Cards)
        cols_per_row = 3
        for row_start in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for c_idx in range(cols_per_row):
                item_idx = row_start + c_idx
                if item_idx < len(recs):
                    item = recs[item_idx]
                    views_fmt = f"{item['views'] / 1e6:.1f}M" if item['views'] >= 1e6 else f"{item['views']:,}"
                    desc_snippet = (item['description'][:135] + "...") if len(str(item['description'])) > 135 else item['description']

                    tags_raw = ast.literal_eval(item['tags']) if isinstance(item['tags'], str) else item['tags']
                    tag_spans = []
                    for t in tags_raw[:5]:
                        if t in item['shared_tags']:
                            tag_spans.append(f"<span style='background:#FEF2F2; color:#DC2626; border:1px solid #FCA5A5; font-size:0.72rem; font-weight:700; padding:2px 7px; border-radius:5px;'>✓ {t}</span>")
                        else:
                            tag_spans.append(f"<span style='background:#F8FAFC; color:#475569; border:1px solid #E2E8F0; font-size:0.72rem; font-weight:500; padding:2px 7px; border-radius:5px;'>{t}</span>")
                    tags_html = " ".join(tag_spans)

                    # Explainability text
                    if item['shared_tag_count'] > 0:
                        matched_str = ", ".join(item['shared_tags'][:3])
                        explain_text = f"<b>Why recommended:</b> Shares <b>{item['shared_tag_count']} core topics</b> with your talk (<i>{matched_str}</i>)."
                    else:
                        explain_text = "<b>Why recommended:</b> Discovered via <b>latent viewer affinity</b> across audience viewing patterns."

                    with cols[c_idx]:
                        st.markdown(f"""
                            <div class="rec-card-box">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <span style="background:#F1F5F9; color:#475569; font-weight:700; font-size:0.76rem; padding:3px 8px; border-radius:6px; border:1px solid #E2E8F0;">#{item_idx + 1}</span>
                                    <span style="background:#FEF2F2; color:#DC2626; border:1px solid #FCA5A5; font-weight:800; font-size:0.78rem; padding:3px 10px; border-radius:9999px;">{item['match_score']:.1f}% Match</span>
                                </div>
                                <div style="font-size:1.05rem; font-weight:800; color:#0F172A; line-height:1.35; margin-bottom:6px; min-height:46px;">
                                    {item['title']}
                                </div>
                                <div style="color:#475569; font-size:0.88rem; font-weight:600; margin-bottom:8px;">
                                    🎙️ {item['main_speaker']}
                                </div>
                                <div style="display:flex; gap:14px; color:#64748B; font-size:0.8rem; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #F1F5F9;">
                                    <span>⏱️ {item['duration_minutes']} min</span>
                                    <span>👁️ {views_fmt} views</span>
                                </div>
                                <div style="color:#334155; font-size:0.84rem; line-height:1.45; margin-bottom:12px; min-height:55px;">
                                    {desc_snippet}
                                </div>
                                <div style="background:#F8FAFC; border-left:3.5px solid #E62B1E; border-radius:0 8px 8px 0; padding:8px 12px; font-size:0.78rem; color:#475569; margin-bottom:12px; line-height:1.45;">
                                    {explain_text}
                                </div>
                                <div style="display:flex; flex-wrap:wrap; gap:5px; margin-bottom:14px;">
                                    {tags_html}
                                </div>
                                <a class="watch-button" href="{item['url']}" target="_blank">▶ Watch on TED.com</a>
                            </div>
                        """, unsafe_allow_html=True)
                        st.write("")


# ------------------------------------------------------------------------------
# TAB 2: DISCOVER BY TOPICS & TAGS
# ------------------------------------------------------------------------------
with tab_topic:
    st.markdown("<div style='font-size:0.96rem; font-weight:700; color:#0F172A; margin-bottom:8px;'>Explore curated talks by selecting topics of interest:</div>", unsafe_allow_html=True)

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
                    desc_snippet = (item['description'][:135] + "...") if len(str(item['description'])) > 135 else item['description']
                    tags_raw = ast.literal_eval(item['tags']) if isinstance(item['tags'], str) else item['tags']

                    tag_spans = []
                    for t in tags_raw[:5]:
                        if t in item['shared_tags']:
                            tag_spans.append(f"<span style='background:#FEF2F2; color:#DC2626; border:1px solid #FCA5A5; font-size:0.72rem; font-weight:700; padding:2px 7px; border-radius:5px;'>✓ {t}</span>")
                        else:
                            tag_spans.append(f"<span style='background:#F8FAFC; color:#475569; border:1px solid #E2E8F0; font-size:0.72rem; font-weight:500; padding:2px 7px; border-radius:5px;'>{t}</span>")
                    tags_html = " ".join(tag_spans)

                    with cols[c_idx]:
                        st.markdown(f"""
                            <div class="rec-card-box">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <span style="background:#F1F5F9; color:#475569; font-weight:700; font-size:0.76rem; padding:3px 8px; border-radius:6px; border:1px solid #E2E8F0;">#{item_idx + 1}</span>
                                    <span style="background:#FEF2F2; color:#DC2626; border:1px solid #FCA5A5; font-weight:800; font-size:0.78rem; padding:3px 10px; border-radius:9999px;">{item['match_score']:.1f}% Relevance</span>
                                </div>
                                <div style="font-size:1.05rem; font-weight:800; color:#0F172A; line-height:1.35; margin-bottom:6px; min-height:46px;">
                                    {item['title']}
                                </div>
                                <div style="color:#475569; font-size:0.88rem; font-weight:600; margin-bottom:8px;">
                                    🎙️ {item['main_speaker']}
                                </div>
                                <div style="display:flex; gap:14px; color:#64748B; font-size:0.8rem; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #F1F5F9;">
                                    <span>⏱️ {item['duration_minutes']} min</span>
                                    <span>👁️ {views_fmt} views</span>
                                </div>
                                <div style="color:#334155; font-size:0.84rem; line-height:1.45; margin-bottom:12px; min-height:55px;">
                                    {desc_snippet}
                                </div>
                                <div style="background:#F8FAFC; border-left:3.5px solid #E62B1E; border-radius:0 8px 8px 0; padding:8px 12px; font-size:0.78rem; color:#475569; margin-bottom:12px; line-height:1.45;">
                                    <b>Topic Fit:</b> Matches <b>{item['shared_tag_count']}</b> of your selected category tags.
                                </div>
                                <div style="display:flex; flex-wrap:wrap; gap:5px; margin-bottom:14px;">
                                    {tags_html}
                                </div>
                                <a class="watch-button" href="{item['url']}" target="_blank">▶ Watch on TED.com</a>
                            </div>
                        """, unsafe_allow_html=True)
                        st.write("")
    else:
        st.info("Please select at least one topic tag above to explore talks.")


# ------------------------------------------------------------------------------
# TAB 3: ML MODEL INSIGHTS & BENCHMARKS (MODULE 5 SYNERGY)
# ------------------------------------------------------------------------------
with tab_insights:
    st.markdown("<div style='font-size:1.25rem; font-weight:850; color:#0F172A; margin-bottom:6px;'>📊 Module 5: Recommendation Quality Benchmark</div>", unsafe_allow_html=True)
    st.markdown("""
        To guarantee recommendation fidelity, the recommendation engine was evaluated against **257 validated test queries**
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
