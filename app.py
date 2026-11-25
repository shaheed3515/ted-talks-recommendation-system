"""
Personalized TED Talks Recommendation System
Production Interactive Web Application (Streamlit)
Infosys Springboard AI/ML Internship Program
Developer: Shaheed Shaik
"""

import os
import ast
import html
import random
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
    page_title="TED Talks Recommender",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CLEAN INLINE SVG ICONS (No Clipart Emoji Clutter)
# ==============================================================================
SVG_CLOCK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
SVG_EYE = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>'
SVG_SPARKLE = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>'
SVG_PLAY = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><polygon points="6 3 20 12 6 21 6 3"/></svg>'
SVG_MIC = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>'


# ==============================================================================
# 3. MODERN EDITORIAL DESIGN SYSTEM & ACCESSIBLE MOTION CSS
# ==============================================================================
def inject_design_system():
    """Injects production-grade typography, card grid, shadows, and accessible motion."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-canvas: #F9F9FB;
        --bg-surface: #FFFFFF;
        --bg-subtle: #F3F4F6;
        --ted-red: #E62B1E;
        --ted-red-dark: #C41E14;
        --ted-red-tint: #FEF2F2;
        --ink-primary: #0F172A;
        --ink-secondary: #475569;
        --ink-muted: #64748B;
        --border-subtle: #E2E8F0;
        --border-strong: #CBD5E1;
        --in-oklab: ;
    }

    @supports (background: linear-gradient(in oklab, white, black)) {
        :root {
            --in-oklab: in oklab;
        }
    }

    /* Global Typography & Base Surface */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: var(--ink-primary) !important;
    }

    .stApp {
        background-color: var(--bg-canvas) !important;
    }

    /* Transparent Click-Through Header + Preserved Sidebar Toggle */
    header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        height: auto !important;
        pointer-events: none !important;
        z-index: 9999 !important;
    }
    [data-testid="stToolbar"] {
        background: transparent !important;
        pointer-events: none !important;
    }
    [data-testid="stExpandSidebarButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
        margin: 8px 0 0 10px !important;
        padding: 4px !important;
        transition: transform 0.18s ease, border-color 0.18s ease, background-color 0.18s ease !important;
    }
    [data-testid="stExpandSidebarButton"]:hover {
        background-color: var(--ted-red-tint) !important;
        border-color: var(--ted-red) !important;
        transform: scale(1.05) !important;
    }
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stExpandSidebarButton"] span {
        pointer-events: auto !important;
        color: var(--ted-red) !important;
        fill: var(--ted-red) !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }
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

    /* Main Container Layout */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1220px !important;
    }

    /* Sidebar Surface & Bottom-Anchored Footer */
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebarUserContent"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: calc(100vh - 4rem) !important;
        padding-bottom: 1.5rem !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    div[role="radiogroup"] label div,
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span {
        color: var(--ink-primary) !important;
        font-weight: 500 !important;
    }
    .sidebar-footer-card {
        margin-top: auto !important;
        background: var(--bg-canvas) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        padding: 0.875rem 1rem !important;
        text-align: center !important;
    }

    /* Crisp Input & Select Controls (Overriding Browser Dark Mode Autofill) */
    input,
    textarea,
    .stTextInput input,
    .stSelectbox input,
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        color: var(--ink-primary) !important;
        background-color: var(--bg-surface) !important;
        caret-color: var(--ted-red) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        -webkit-text-fill-color: var(--ink-primary) !important;
        opacity: 1 !important;
    }
    .stTextInput div[data-baseweb="input"] {
        background-color: var(--bg-surface) !important;
        border: 1.5px solid var(--border-strong) !important;
        border-radius: 12px !important;
        padding: 0.15rem 0.5rem !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    }
    .stTextInput div[data-baseweb="input"]:focus-within {
        border-color: var(--ted-red) !important;
        box-shadow: 0 0 0 3px rgba(230, 43, 30, 0.14) !important;
    }
    .stTextInput input::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--bg-surface) !important;
        border: 1.5px solid var(--border-strong) !important;
        border-radius: 10px !important;
        color: var(--ink-primary) !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: var(--ink-primary) !important;
        font-weight: 600 !important;
    }

    /* Native Buttons Polish */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        min-height: 2.65rem !important;
        border: 1px solid var(--border-strong) !important;
        background-color: var(--bg-surface) !important;
        color: var(--ink-primary) !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--ted-red) !important;
        color: var(--ted-red) !important;
        background-color: var(--ted-red-tint) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--ted-red) !important;
        border-color: var(--ted-red) !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--ted-red-dark) !important;
        border-color: var(--ted-red-dark) !important;
        color: #FFFFFF !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        border-bottom: 1.5px solid var(--border-subtle) !important;
        margin-bottom: 1.25rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        color: var(--ink-muted) !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.1rem !important;
        transition: color 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--ink-primary) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--ted-red) !important;
        font-weight: 700 !important;
        border-bottom-color: var(--ted-red) !important;
    }

    /* App Editorial Header */
    .app-hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .ted-badge {
        background-color: var(--ted-red);
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.45rem;
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        letter-spacing: -0.04em;
        box-shadow: 0 4px 12px rgba(230, 43, 30, 0.22);
    }
    .brand-title {
        font-size: clamp(1.35rem, 1.1rem + 1vw, 1.85rem);
        font-weight: 800;
        color: var(--ink-primary);
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
        text-wrap: balance;
    }
    .brand-subtitle {
        color: var(--ink-secondary);
        font-size: 0.92rem;
        font-weight: 500;
        margin: 0.15rem 0 0 0;
        line-height: 1.45;
    }
    .catalog-pill {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 9999px;
        padding: 0.4rem 0.9rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--ink-secondary);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    /* Seed / Currently Exploring Hero Card */
    .seed-hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF8F7 100%);
        background: linear-gradient(135deg var(--in-oklab), #FFFFFF 0%, #FFF8F7 100%);
        border: 1px solid var(--border-subtle);
        border-left: 5px solid var(--ted-red);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0 1.75rem 0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 10px 25px -5px rgba(15, 23, 42, 0.06);
    }
    .seed-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: var(--ted-red-tint);
        color: var(--ted-red-dark);
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .seed-title {
        font-size: clamp(1.2rem, 1rem + 0.6vw, 1.5rem);
        font-weight: 800;
        color: var(--ink-primary);
        margin: 0.65rem 0 0.35rem 0;
        line-height: 1.3;
        text-wrap: balance;
    }
    .seed-meta {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 1rem;
        color: var(--ink-secondary);
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .seed-meta-item {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .seed-desc {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.6;
        max-width: 75ch;
        text-wrap: pretty;
        margin-bottom: 1rem;
    }

    /* Responsive Equal-Height Card Grid */
    .talk-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(315px, 1fr));
        gap: 1.25rem;
        margin: 0.75rem 0 1.5rem 0;
    }
    .talk-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.35rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.85rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 16px -4px rgba(15, 23, 42, 0.05);
        transition: transform 0.24s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.24s ease, border-color 0.24s ease;
    }
    .talk-card:hover {
        transform: translateY(-4px);
        border-color: rgba(230, 43, 30, 0.45);
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06), 0 18px 30px -8px rgba(230, 43, 30, 0.14);
    }

    /* Compositor-Friendly Entrance Animation Opt-In */
    @keyframes cardFadeSlideUp {
        from {
            opacity: 0;
            transform: translateY(14px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @media (prefers-reduced-motion: no-preference) {
        .talk-card {
            animation: cardFadeSlideUp 0.42s cubic-bezier(0.22, 1, 0.36, 1) both;
            animation-delay: calc(var(--card-idx, 0) * 45ms);
        }
        .seed-hero-card {
            animation: cardFadeSlideUp 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
        }
    }

    .card-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
    }
    .rank-chip {
        background: var(--bg-subtle);
        color: var(--ink-secondary);
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
    }
    .match-pill {
        background: var(--ted-red-tint);
        color: var(--ted-red-dark);
        border: 1px solid #FECACA;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
    }
    .card-title {
        font-size: 1.06rem;
        font-weight: 750;
        color: var(--ink-primary);
        line-height: 1.35;
        margin: 0.4rem 0 0.25rem 0;
        text-wrap: balance;
    }
    .card-speaker {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: var(--ink-secondary);
        font-size: 0.86rem;
        font-weight: 600;
    }
    .card-stats {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: var(--ink-muted);
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.5rem 0;
        border-bottom: 1px solid #F1F5F9;
    }
    .card-stat-item {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .card-summary {
        color: #334155;
        font-size: 0.85rem;
        line-height: 1.5;
        text-wrap: pretty;
        margin: 0;
    }
    .explain-strip {
        background: #F8FAFC;
        border-left: 3px solid var(--ted-red);
        border-radius: 0 8px 8px 0;
        padding: 0.55rem 0.75rem;
        font-size: 0.79rem;
        color: var(--ink-secondary);
        line-height: 1.45;
    }
    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    .tag-matched {
        background: var(--ted-red-tint);
        color: var(--ted-red-dark);
        border: 1px solid #FECACA;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.16rem 0.5rem;
        border-radius: 6px;
    }
    .tag-normal {
        background: var(--bg-canvas);
        color: var(--ink-secondary);
        border: 1px solid var(--border-subtle);
        font-size: 0.72rem;
        font-weight: 500;
        padding: 0.16rem 0.5rem;
        border-radius: 6px;
    }
    .watch-cta {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.45rem;
        width: 100%;
        background-color: var(--ted-red);
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 0.86rem;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        text-decoration: none !important;
        box-shadow: 0 2px 6px rgba(230, 43, 30, 0.22);
        transition: background-color 0.18s ease, transform 0.15s ease, box-shadow 0.18s ease;
        box-sizing: border-box;
    }
    .watch-cta:hover {
        background-color: var(--ted-red-dark);
        color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(230, 43, 30, 0.28);
    }
    .watch-cta-inline {
        width: auto;
        padding: 0.5rem 1.15rem;
    }
    .section-heading {
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--ink-primary);
        letter-spacing: -0.015em;
        margin: 0.25rem 0 0.75rem 0;
    }
</style>
""", unsafe_allow_html=True)


inject_design_system()


# ==============================================================================
# 4. CACHED IN-MEMORY MODEL & DATASET INGESTION
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_recommender_system():
    """Loads cleaned talk metadata, TF-IDF sparse vectors, and SVD collaborative factors."""
    data_path = "data/processed/ted_talks_cleaned.csv"
    model_dir = "models"

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please verify project paths.")
        st.stop()

    df = pd.read_csv(data_path)
    df["url"] = df["url"].astype(str).str.strip()

    tags_list = []
    for raw_t in df["tags"]:
        try:
            tags_list.append(set(ast.literal_eval(raw_t)))
        except Exception:
            tags_list.append(set())
    df["parsed_tags"] = tags_list

    tfidf_matrix = sparse.load_npz(os.path.join(model_dir, "tfidf_matrix.npz"))
    user_item_matrix = sparse.load_npz(os.path.join(model_dir, "user_item_matrix.npz"))

    sim_content = cosine_similarity(tfidf_matrix, tfidf_matrix)

    svd = TruncatedSVD(n_components=20, random_state=42)
    svd.fit(user_item_matrix)
    item_factors = svd.components_.T
    sim_collab = cosine_similarity(item_factors, item_factors)

    title_to_idx = {str(title).lower().strip(): idx for idx, title in enumerate(df["title"])}
    popular_tags = sorted({tag for tag_set in tags_list for tag in tag_set})

    return df, sim_content, sim_collab, title_to_idx, popular_tags


with st.spinner("Loading recommendation engine..."):
    df, sim_content, sim_collab, title_to_idx, popular_tags = load_recommender_system()


# ==============================================================================
# 5. CORE INFERENCE, SEARCH & PRESENTATION HELPERS
# ==============================================================================
def format_views(views: int) -> str:
    """Formats view counts cleanly (e.g., 47.2M or 840K)."""
    if views >= 1_000_000:
        return f"{views / 1e6:.1f}M"
    if views >= 1_000:
        return f"{views / 1e3:.0f}K"
    return f"{views:,}"


def parse_tags_list(raw_tags) -> list:
    """Safely parses tag collection into an ordered list."""
    if isinstance(raw_tags, list):
        return raw_tags
    if isinstance(raw_tags, set):
        return sorted(list(raw_tags))
    try:
        return list(ast.literal_eval(str(raw_tags)))
    except Exception:
        return []


def find_matching_talks(query: str, limit: int = 25) -> list:
    """Searches across talk titles, speaker names, and tags in relevance order."""
    if not query or not str(query).strip():
        return []
    q = str(query).strip().lower()

    title_hits = df[df["title"].str.lower().str.contains(q, na=False, regex=False)]
    speaker_hits = df[df["main_speaker"].str.lower().str.contains(q, na=False, regex=False)]
    tag_hits = df[df["tags"].str.lower().str.contains(q, na=False, regex=False)]

    combined = pd.concat([title_hits, speaker_hits, tag_hits]).drop_duplicates(subset=["title"])
    return list(combined["title"].head(limit))


def sort_recommendations(recs: list, sort_by: str) -> list:
    """Applies clean user-chosen ordering while keeping relevance scores intact."""
    if sort_by == "Most Viewed":
        return sorted(recs, key=lambda r: r.get("views", 0), reverse=True)
    if sort_by == "Shortest First":
        return sorted(recs, key=lambda r: r.get("duration_minutes", 999))
    return recs


def passes_duration_filter(duration_minutes: float, duration_filter: str) -> bool:
    """Checks if a talk satisfies the user's selected duration filter."""
    if duration_filter == "Quick (< 10 min)":
        return duration_minutes < 10
    if duration_filter == "Standard (10–18 min)":
        return 10 <= duration_minutes <= 18
    if duration_filter == "Deep Dive (> 18 min)":
        return duration_minutes > 18
    return True


def get_hybrid_recommendations(
    query_title: str,
    strategy: str = "Hybrid (Recommended)",
    alpha: float = 0.70,
    top_n: int = 6,
    duration_filter: str = "All Durations",
    sort_by: str = "Best Match"
):
    """Generates normalized, ranked hybrid recommendations with explainability metadata."""
    q = query_title.lower().strip()
    if q not in title_to_idx:
        matches = [t for t in title_to_idx.keys() if q in t]
        if not matches:
            return None, []
        q = matches[0]

    idx = title_to_idx[q]
    query_talk = df.iloc[idx]
    query_tags = query_talk["parsed_tags"]

    s_c = sim_content[idx].copy()
    s_c[idx] = -1e9
    c_min, c_max = s_c[s_c > -1e8].min(), s_c.max()
    s_c_norm = (s_c - c_min) / (c_max - c_min + 1e-9)
    s_c_norm[idx] = -1e9

    s_cf = sim_collab[idx].copy()
    s_cf[idx] = -1e9
    cf_min, cf_max = s_cf[s_cf > -1e8].min(), s_cf.max()
    s_cf_norm = (s_cf - cf_min) / (cf_max - cf_min + 1e-9)
    s_cf_norm[idx] = -1e9

    if strategy == "Content-Based (TF-IDF)":
        final_scores = s_c_norm
    elif strategy == "Collaborative Filtering (SVD)":
        final_scores = s_cf_norm
    else:
        final_scores = alpha * s_c_norm + (1.0 - alpha) * s_cf_norm

    ranked_indices = np.argsort(final_scores)[::-1]

    results = []
    for candidate_idx in ranked_indices:
        if candidate_idx == idx:
            continue
        row = df.iloc[candidate_idx]
        if not passes_duration_filter(row["duration_minutes"], duration_filter):
            continue

        shared_tags = sorted(list(query_tags.intersection(row["parsed_tags"])))
        row_dict = row.to_dict()
        row_dict["match_score"] = float(np.clip(final_scores[candidate_idx], 0.0, 1.0)) * 100.0
        row_dict["shared_tags"] = shared_tags
        row_dict["shared_tag_count"] = len(shared_tags)
        if shared_tags:
            matched_str = ", ".join(html.escape(t) for t in shared_tags[:3])
            row_dict["reason_html"] = f"Shares <strong>{len(shared_tags)} core topics</strong>: <em>{matched_str}</em>"
        else:
            row_dict["reason_html"] = "Discovered via <strong>latent audience affinity</strong> across viewer watch history"
        results.append(row_dict)

        if len(results) >= top_n:
            break

    return query_talk, sort_recommendations(results, sort_by)


def get_recommendations_by_tags(
    selected_tags: list,
    top_n: int = 6,
    duration_filter: str = "All Durations",
    sort_by: str = "Best Match"
) -> list:
    """Scores talks by Jaccard topic overlap blended with log-scaled popularity."""
    if not selected_tags:
        return []
    target_set = set(selected_tags)
    scored = []

    for idx, row in df.iterrows():
        if not passes_duration_filter(row["duration_minutes"], duration_filter):
            continue
        overlap = len(target_set.intersection(row["parsed_tags"]))
        if overlap == 0:
            continue
        jaccard = overlap / float(len(target_set.union(row["parsed_tags"])))
        score = jaccard * 0.72 + (np.log1p(row["views"]) / 20.0) * 0.28
        scored.append((idx, score, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for idx, score, overlap in scored[:top_n]:
        row_dict = df.iloc[idx].to_dict()
        shared = sorted(list(target_set.intersection(row_dict["parsed_tags"])))
        row_dict["match_score"] = min(99.0, score * 115.0)
        row_dict["shared_tags"] = shared
        row_dict["shared_tag_count"] = overlap
        row_dict["reason_html"] = f"Matches <strong>{overlap}</strong> of your selected topics"
        results.append(row_dict)

    return sort_recommendations(results, sort_by)


def render_featured_hero(query_talk):
    """Renders the selected seed talk in an editorial hero card."""
    title_safe = html.escape(str(query_talk["title"]))
    speaker_safe = html.escape(str(query_talk["main_speaker"]))
    occupation = str(query_talk.get("speaker_occupation", "") or "").strip()
    occ_safe = f" — {html.escape(occupation)}" if occupation and occupation.lower() != "nan" else ""
    desc_safe = html.escape(str(query_talk["description"]))
    views_str = format_views(int(query_talk["views"]))
    duration_str = f"{query_talk['duration_minutes']:.1f} min" if isinstance(query_talk["duration_minutes"], float) else f"{query_talk['duration_minutes']} min"
    url_safe = html.escape(str(query_talk["url"]))

    tags_raw = parse_tags_list(query_talk["tags"])
    tags_html = "".join(
        f"<span class='tag-normal'>{html.escape(str(t))}</span>" for t in tags_raw[:6]
    )

    st.markdown(f"""
        <article class="seed-hero-card">
            <div class="seed-eyebrow">{SVG_SPARKLE} Selected Benchmark Talk</div>
            <h2 class="seed-title">{title_safe}</h2>
            <div class="seed-meta">
                <span class="seed-meta-item">{SVG_MIC} <strong>{speaker_safe}</strong>{occ_safe}</span>
                <span class="seed-meta-item">{SVG_CLOCK} {duration_str}</span>
                <span class="seed-meta-item">{SVG_EYE} {views_str} views</span>
            </div>
            <p class="seed-desc">{desc_safe}</p>
            <div class="tag-row" style="margin-bottom: 1.1rem;">{tags_html}</div>
            <a class="watch-cta watch-cta-inline" href="{url_safe}" target="_blank" rel="noopener noreferrer">
                {SVG_PLAY} Watch on TED.com
            </a>
        </article>
    """, unsafe_allow_html=True)


def render_talk_cards_grid(recs: list, score_label: str = "Match", empty_msg: str = "No talks match the active filters."):
    """Renders an accessible, animated responsive CSS Grid of talk cards without duplicate logic."""
    if not recs:
        st.info(empty_msg)
        return

    cards_html = []
    for idx, item in enumerate(recs):
        title_safe = html.escape(str(item["title"]))
        speaker_safe = html.escape(str(item["main_speaker"]))
        raw_desc = str(item["description"])
        desc_trim = (raw_desc[:140] + "...") if len(raw_desc) > 140 else raw_desc
        desc_safe = html.escape(desc_trim)
        views_str = format_views(int(item["views"]))
        dur_val = item["duration_minutes"]
        dur_str = f"{dur_val:.1f} min" if isinstance(dur_val, float) else f"{dur_val} min"
        url_safe = html.escape(str(item["url"]))

        tags_raw = parse_tags_list(item["tags"])
        shared_set = set(item.get("shared_tags", []))
        tag_spans = []
        for t in tags_raw[:4]:
            t_safe = html.escape(str(t))
            cls = "tag-matched" if t in shared_set else "tag-normal"
            tag_spans.append(f"<span class='{cls}'>{t_safe}</span>")
        tags_html = "".join(tag_spans)

        score_badge = f"{item['match_score']:.1f}% {html.escape(score_label)}" if "match_score" in item else "Saved"
        reason_block = f"<div class='explain-strip'>{item['reason_html']}</div>" if "reason_html" in item else ""

        cards_html.append(f"""
        <article class="talk-card" style="--card-idx: {idx};">
            <div>
                <div class="card-top-row">
                    <span class="rank-chip">#{idx + 1}</span>
                    <span class="match-pill">{score_badge}</span>
                </div>
                <h3 class="card-title">{title_safe}</h3>
                <div class="card-speaker">{SVG_MIC} {speaker_safe}</div>
                <div class="card-stats">
                    <span class="card-stat-item">{SVG_CLOCK} {dur_str}</span>
                    <span class="card-stat-item">{SVG_EYE} {views_str} views</span>
                </div>
            </div>
            <p class="card-summary">{desc_safe}</p>
            {reason_block}
            <div>
                <div class="tag-row" style="margin-bottom: 0.9rem;">{tags_html}</div>
                <a class="watch-cta" href="{url_safe}" target="_blank" rel="noopener noreferrer">
                    {SVG_PLAY} Watch on TED.com
                </a>
            </div>
        </article>
        """)

    st.markdown(f'<section class="talk-grid">{"".join(cards_html)}</section>', unsafe_allow_html=True)


# ==============================================================================
# 6. SESSION STATE INITIALIZATION
# ==============================================================================
if "selected_talk" not in st.session_state:
    st.session_state.selected_talk = "Do schools kill creativity?"
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []


# ==============================================================================
# 7. SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; padding-bottom:14px; margin-bottom:14px; border-bottom:1px solid #E2E8F0;">
            <span style="background:#E62B1E; color:#FFFFFF; font-weight:800; font-size:1.15rem; padding:3px 9px; border-radius:6px; letter-spacing:-0.5px;">TED</span>
            <span style="font-size:1.08rem; font-weight:750; color:#0F172A;">Recommender</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.84rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px;'>Engine Strategy</div>", unsafe_allow_html=True)
    strategy = st.radio(
        "Recommendation Strategy",
        options=["Hybrid (Recommended)", "Content-Based (TF-IDF)", "Collaborative Filtering (SVD)"],
        index=0,
        label_visibility="collapsed"
    )

    alpha_val = 0.70
    if strategy == "Hybrid (Recommended)":
        st.markdown("<div style='margin-top:12px; font-weight:650; font-size:0.85rem; color:#0F172A;'>Content vs. Audience Blend</div>", unsafe_allow_html=True)
        alpha_val = st.slider(
            "Hybrid Alpha",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            label_visibility="collapsed"
        )
        col_c, col_cf = st.columns(2)
        col_c.markdown(f"<span style='font-size:0.78rem; color:#475569;'>Content: <b>{int(alpha_val * 100)}%</b></span>", unsafe_allow_html=True)
        col_cf.markdown(f"<span style='font-size:0.78rem; color:#475569;'>Audience: <b>{int((1.0 - alpha_val) * 100)}%</b></span>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:16px 0; border:none; border-top:1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.84rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px;'>Filters & Sorting</div>", unsafe_allow_html=True)
    top_n_count = st.slider("Results per page", min_value=3, max_value=12, value=6, step=3)

    duration_choice = st.selectbox(
        "Talk duration",
        options=["All Durations", "Quick (< 10 min)", "Standard (10–18 min)", "Deep Dive (> 18 min)"],
        index=0
    )

    sort_choice = st.selectbox(
        "Sort results by",
        options=["Best Match", "Most Viewed", "Shortest First"],
        index=0
    )

    st.markdown("""
        <div class="sidebar-footer-card">
            <div style="font-size:0.72rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:3px;">
                Infosys Springboard
            </div>
            <div style="font-size:0.84rem; font-weight:750; color:#0F172A; margin-bottom:3px;">
                AI/ML Internship Project
            </div>
            <div style="font-size:0.8rem; font-weight:600; color:#475569;">
                Developer: <b style="color:#E62B1E;">Shaheed Shaik</b>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 8. EDITORIAL APPLICATION HEADER
# ==============================================================================
st.markdown(f"""
    <header class="app-hero">
        <div class="brand-lockup">
            <span class="ted-badge">TED</span>
            <div>
                <h1 class="brand-title">Talks Discovery Engine</h1>
                <p class="brand-subtitle">Personalized recommendations powered by Hybrid NLP &amp; Matrix Factorization</p>
            </div>
        </div>
        <div class="catalog-pill">{len(df):,} Curated Talks</div>
    </header>
""", unsafe_allow_html=True)


# ==============================================================================
# 9. NAVIGATION TABS
# ==============================================================================
watchlist_count = len(st.session_state.watchlist)
tab_discover, tab_topics, tab_watchlist, tab_insights = st.tabs([
    "Discover Talks",
    "Topic Collections",
    f"Saved Watchlist ({watchlist_count})",
    "Model Insights"
])


def _select_talk_cb(talk_title: str):
    """Callback to switch active seed talk cleanly before widget re-instantiation."""
    st.session_state.selected_talk = talk_title
    st.session_state.search_bar_input = ""


def _shuffle_talk_cb():
    """Callback to pick a random talk cleanly before widget re-instantiation."""
    st.session_state.selected_talk = random.choice(list(df["title"]))
    st.session_state.search_bar_input = ""


def _pivot_talk_cb():
    """Callback to pivot to a recommended talk from the dropdown."""
    choice = st.session_state.get("pivot_select", "Explore a recommendation...")
    if choice != "Explore a recommendation...":
        st.session_state.selected_talk = choice
        st.session_state.search_bar_input = ""
        st.session_state.pivot_select = "Explore a recommendation..."


def _set_topic_preset_cb(tags_list: list):
    """Callback to switch topic collection tags cleanly."""
    st.session_state.topic_selection = tags_list


# ------------------------------------------------------------------------------
# TAB 1: DISCOVER TALKS
# ------------------------------------------------------------------------------
with tab_discover:
    col_search, col_shuffle = st.columns([4.5, 1])

    with col_search:
        search_val = st.text_input(
            "Search talks",
            placeholder="Search by talk title, speaker name, or topic (e.g., Ken Robinson, creativity, neuroscience)...",
            label_visibility="collapsed",
            key="search_bar_input"
        )

    with col_shuffle:
        st.button("Surprise Me", type="primary", width="stretch", on_click=_shuffle_talk_cb)

    # Handle Search Query Matches Cleanly
    query_str = search_val.strip() if search_val else ""
    if query_str:
        matched_talks = find_matching_talks(query_str, limit=20)
        if matched_talks:
            if len(matched_talks) == 1:
                if st.session_state.selected_talk != matched_talks[0]:
                    st.session_state.selected_talk = matched_talks[0]
                    st.rerun()
            else:
                sel_idx = matched_talks.index(st.session_state.selected_talk) if st.session_state.selected_talk in matched_talks else 0
                chosen_match = st.selectbox(
                    f"Found {len(matched_talks)} talks matching \"{query_str}\":",
                    options=matched_talks,
                    index=sel_idx,
                    key="search_results_dropdown"
                )
                if chosen_match != st.session_state.selected_talk:
                    st.session_state.selected_talk = chosen_match
                    st.rerun()
                elif st.session_state.selected_talk not in matched_talks:
                    st.session_state.selected_talk = matched_talks[0]
                    st.rerun()
        else:
            st.info(f"No talks matched \"{query_str}\". Try searching a speaker or topic, or pick a featured talk below.")

    # Concise Quick-Pick Chips (Zero Title Truncation)
    quick_picks = [
        ("Creativity", "Do schools kill creativity?"),
        ("Body Language", "Your body language may shape who you are"),
        ("Vulnerability", "The power of vulnerability"),
        ("Procrastination", "Inside the mind of a master procrastinator"),
        ("Leadership", "How great leaders inspire action"),
    ]
    pick_cols = st.columns(len(quick_picks))
    for i, (label, full_title) in enumerate(quick_picks):
        with pick_cols[i]:
            st.button(label, key=f"quick_{i}", width="stretch", on_click=_select_talk_cb, args=(full_title,))

    # Run Inference
    query_talk, recs = get_hybrid_recommendations(
        query_title=st.session_state.selected_talk,
        strategy=strategy,
        alpha=alpha_val,
        top_n=top_n_count,
        duration_filter=duration_choice,
        sort_by=sort_choice
    )

    if query_talk is not None:
        render_featured_hero(query_talk)

        # Bookmark Action Bar for Featured Talk
        current_title = str(query_talk["title"])
        is_saved = current_title in st.session_state.watchlist
        col_head, col_save, col_pivot = st.columns([3.2, 1.2, 1.6])
        with col_head:
            st.markdown(f"<div class='section-heading'>Recommended Next ({len(recs)})</div>", unsafe_allow_html=True)
        with col_save:
            btn_label = "Saved to Watchlist ✓" if is_saved else "+ Save Current Talk"
            if st.button(btn_label, key="toggle_watchlist_btn", width="stretch"):
                if is_saved:
                    st.session_state.watchlist.remove(current_title)
                else:
                    st.session_state.watchlist.append(current_title)
                st.rerun()
        with col_pivot:
            if recs:
                pivot_options = ["Explore a recommendation..."] + [r["title"] for r in recs]
                st.selectbox(
                    "Explore next",
                    options=pivot_options,
                    index=0,
                    label_visibility="collapsed",
                    key="pivot_select",
                    on_change=_pivot_talk_cb
                )

        render_talk_cards_grid(recs, score_label="Match")


# ------------------------------------------------------------------------------
# TAB 2: TOPIC COLLECTIONS
# ------------------------------------------------------------------------------
with tab_topics:
    st.markdown("<div class='section-heading'>Explore Curated Topic Collections</div>", unsafe_allow_html=True)

    curated_collections = {
        "AI & Future Tech": ["technology", "AI", "robots", "future"],
        "Mind & Psychology": ["psychology", "brain", "happiness", "mental health"],
        "Design & Creativity": ["creativity", "design", "art", "innovation"],
        "Leadership & Growth": ["leadership", "motivation", "success", "business"],
    }

    if "topic_selection" not in st.session_state:
        st.session_state.topic_selection = ["creativity", "education"]

    preset_cols = st.columns(len(curated_collections))
    for i, (preset_name, preset_tags) in enumerate(curated_collections.items()):
        valid_preset = [t for t in preset_tags if t in popular_tags]
        with preset_cols[i]:
            st.button(preset_name, key=f"preset_{i}", width="stretch", on_click=_set_topic_preset_cb, args=(valid_preset,))

    selected_tags = st.multiselect(
        "Customize topics",
        options=popular_tags,
        key="topic_selection",
        label_visibility="collapsed"
    )

    if selected_tags:
        topic_recs = get_recommendations_by_tags(
            selected_tags,
            top_n=top_n_count,
            duration_filter=duration_choice,
            sort_by=sort_choice
        )
        render_talk_cards_grid(
            topic_recs,
            score_label="Relevance",
            empty_msg="No talks matched all active filters. Try relaxing the duration filter."
        )
    else:
        st.info("Select a curated collection or add topic tags above to explore talks.")


# ------------------------------------------------------------------------------
# TAB 3: SAVED WATCHLIST
# ------------------------------------------------------------------------------
with tab_watchlist:
    col_wl_title, col_wl_clear = st.columns([4, 1])
    with col_wl_title:
        st.markdown("<div class='section-heading'>Your Personal Watchlist</div>", unsafe_allow_html=True)
    with col_wl_clear:
        if st.session_state.watchlist and st.button("Clear All", width="stretch"):
            st.session_state.watchlist = []
            st.rerun()

    if st.session_state.watchlist:
        saved_items = []
        for saved_title in st.session_state.watchlist:
            match_rows = df[df["title"] == saved_title]
            if not match_rows.empty:
                saved_items.append(match_rows.iloc[0].to_dict())
        render_talk_cards_grid(saved_items, score_label="Saved")
    else:
        st.info("Your watchlist is empty. Select any talk in 'Discover Talks' and click '+ Save Current Talk' to build your personal queue.")


# ------------------------------------------------------------------------------
# TAB 4: MODEL INSIGHTS & BENCHMARKS (MODULE 5)
# ------------------------------------------------------------------------------
with tab_insights:
    st.markdown("<div class='section-heading'>Model Architecture &amp; Offline Evaluation Benchmarks</div>", unsafe_allow_html=True)
    st.markdown("""
        Evaluated across **257 validated benchmark queries** using Information Retrieval (IR) metrics.
        The **Hybrid Engine (α = 0.70)** balances high topical accuracy with cross-domain diversity and zero cold-start failures.
    """)

    benchmark_data = {
        "Model Strategy": [
            "Random Baseline",
            "Collaborative Filtering (SVD, k=20)",
            "Content-Based (TF-IDF Cosine)",
            "Hybrid Recommender (α = 0.70) [Production]"
        ],
        "Precision@5": ["0.0389", "0.0389", "0.4296", "0.4156"],
        "Recall@10": ["0.0046", "0.0057", "0.0797", "0.0701"],
        "NDCG@10": ["0.0395", "0.0361", "0.4237", "0.4039"],
        "MAP@10": ["0.0141", "0.0122", "0.2969", "0.2821"],
        "Diversity (ILD)": ["0.9492", "0.9489", "0.8871", "0.8925"],
        "Catalog Coverage": ["63.33%", "62.75%", "57.22%", "57.57%"]
    }
    st.dataframe(pd.DataFrame(benchmark_data), width="stretch", hide_index=True)

    st.markdown("<hr style='margin:1.5rem 0; border:none; border-top:1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        if os.path.exists("docs/figures/11_evaluation_metrics_comparison.png"):
            st.image("docs/figures/11_evaluation_metrics_comparison.png", caption="Figure 11: Ranking Quality Benchmark Comparison", width="stretch")
        if os.path.exists("docs/figures/13_hybrid_alpha_sensitivity.png"):
            st.image("docs/figures/13_hybrid_alpha_sensitivity.png", caption="Figure 13: Alpha Weight Sensitivity Analysis", width="stretch")

    with col_fig2:
        if os.path.exists("docs/figures/12_ndcg_precision_curve.png"):
            st.image("docs/figures/12_ndcg_precision_curve.png", caption="Figure 12: Precision@10 vs. Recall@10 Frontier", width="stretch")
        if os.path.exists("docs/figures/14_model_tradeoff_radar.png"):
            st.image("docs/figures/14_model_tradeoff_radar.png", caption="Figure 14: Multi-Dimensional Trade-Off Radar", width="stretch")
