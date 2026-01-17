"""
UI components for the NBA Stats Predictor app.
Contains reusable components and page layouts.
"""

import streamlit as st
from typing import List, Optional, Callable


def apply_dark_theme():
    """Apply the dark theme CSS to the app."""
    st.markdown("""
    <style>
    /* ===== GLOBAL APP BACKGROUND ===== */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* ===== TEXT ===== */
    html, body, [class*="css"] {
        color: #FAFAFA !important;
    }
    /* ===== HEADERS ===== */
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA;
    }
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #111827;
    }
    [data-testid="stSidebar"] * {
        color: #FAFAFA;
    }
    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background-color: #1F2937;
        padding: 16px;
        border-radius: 10px;
        color: #FAFAFA;
    }
    [data-testid="stMetricLabel"] {
        color: #9CA3AF;
    }
    [data-testid="stMetricValue"] {
        color: #FAFAFA;
        font-size: 1.6rem;
    }
    /* ===== DATAFRAMES ===== */
    .stDataFrame {
        background-color: #1F2937;
    }
    .stDataFrame td, .stDataFrame th {
        color: #FAFAFA !important;
        background-color: #1F2937 !important;
    }
    /* ===== INPUTS ===== */
    input, textarea, select {
        background-color: #1F2937 !important;
        color: #FAFAFA !important;
    }
    /* ===== BUTTONS ===== */
    .stButton > button, .stLinkButton > a {
        background-color: #FF6B35 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .stButton > button:hover, .stLinkButton > a:hover {
        background-color: #FF8C5A !important;
        border-color: #FF8C5A !important;
    }
    /* ===== SUCCESS / INFO / WARNING ===== */
    .stAlert {
        background-color: #1F2937;
        color: #FAFAFA;
    }
    /* ===== PLOTS ===== */
    svg, text {
        fill: #FAFAFA !important;
    }
    /* ===== CUSTOM CARDS ===== */
    .player-card {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #374151;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .player-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.2);
    }
    .team-card {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #374151;
    }
    .section-header {
        color: #FF6B35;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .user-profile {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        background: #1F2937;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .user-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        object-fit: cover;
    }
    .favorite-heart {
        color: #EF4444;
        position: absolute;
        top: 8px;
        right: 8px;
    }
    .stat-badge {
        background: #FF6B35;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .empty-state {
        text-align: center;
        padding: 40px;
        color: #9CA3AF;
    }
    .quick-action-btn {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C5A 100%);
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        color: white;
        font-weight: 600;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)


def render_user_profile_sidebar(user_info: dict, on_logout: Callable):
    """Render user profile in sidebar."""
    picture_url = user_info.get("picture")
    name = user_info.get("name", "User")
    email = user_info.get("email", "")
    
    st.sidebar.markdown("---")
    
    # User info section
    col1, col2 = st.sidebar.columns([1, 3])
    
    with col1:
        if picture_url:
            st.image(picture_url, width=50)
        else:
            st.markdown("👤")
    
    with col2:
        st.markdown(f"**{name}**")
        st.caption(email[:20] + "..." if len(email) > 20 else email)
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        on_logout()


def render_player_card(player_name: str, team: str = "", on_view: Callable = None, on_remove: Callable = None, show_heart: bool = True):
    """Render a player card with avatar and actions."""
    # Generate initials for avatar placeholder
    initials = "".join([n[0].upper() for n in player_name.split()[:2]])
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #374151;
        position: relative;
        margin-bottom: 12px;
    ">
        {"<span style='position: absolute; top: 8px; right: 8px; color: #EF4444; font-size: 1.2rem;'>❤️</span>" if show_heart else ""}
        <div style="
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #FF6B35 0%, #FF8C5A 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 12px auto;
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
        ">{initials}</div>
        <div style="font-weight: 600; font-size: 1rem; color: #FAFAFA; margin-bottom: 4px;">
            {player_name}
        </div>
        <div style="color: #9CA3AF; font-size: 0.85rem; margin-bottom: 12px;">
            {team}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Stats", key=f"view_{player_name}", use_container_width=True):
            if on_view:
                on_view(player_name)
    with col2:
        if st.button("❌", key=f"remove_{player_name}", use_container_width=True):
            if on_remove:
                on_remove(player_name)


def render_team_card(team_abbrev: str, def_rating: float = None, on_remove: Callable = None):
    """Render a team card with logo placeholder and stats."""
    # Team name mapping (abbreviated)
    team_names = {
        "LAL": "Lakers", "GSW": "Warriors", "MIL": "Bucks", "BOS": "Celtics",
        "PHX": "Suns", "MIA": "Heat", "DEN": "Nuggets", "PHI": "76ers",
        "LAC": "Clippers", "DAL": "Mavericks", "MEM": "Grizzlies", "CLE": "Cavaliers",
        "NYK": "Knicks", "BKN": "Nets", "ATL": "Hawks", "CHI": "Bulls",
        "TOR": "Raptors", "SAC": "Kings", "MIN": "Timberwolves", "NOP": "Pelicans",
        "OKC": "Thunder", "POR": "Trail Blazers", "UTA": "Jazz", "IND": "Pacers",
        "WAS": "Wizards", "ORL": "Magic", "CHA": "Hornets", "DET": "Pistons",
        "HOU": "Rockets", "SAS": "Spurs"
    }
    
    team_name = team_names.get(team_abbrev, team_abbrev)
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #374151;
        margin-bottom: 12px;
    ">
        <div style="
            font-size: 2rem;
            margin-bottom: 8px;
        ">🏀</div>
        <div style="font-weight: 600; font-size: 1.1rem; color: #FAFAFA;">
            {team_abbrev}
        </div>
        <div style="color: #9CA3AF; font-size: 0.85rem; margin-bottom: 8px;">
            {team_name}
        </div>
        {f'<div style="background: #FF6B35; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; display: inline-block;">DEF RTG: {def_rating}</div>' if def_rating else ''}
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("❌ Remove", key=f"remove_team_{team_abbrev}", use_container_width=True):
        if on_remove:
            on_remove(team_abbrev)


def render_empty_state(message: str, icon: str = "📭"):
    """Render an empty state placeholder."""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 40px 20px;
        color: #9CA3AF;
        background: #1F2937;
        border-radius: 12px;
        border: 1px dashed #374151;
    ">
        <div style="font-size: 3rem; margin-bottom: 12px;">{icon}</div>
        <div style="font-size: 1rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, icon: str = ""):
    """Render a styled section header."""
    st.markdown(f"""
    <div style="
        color: #FF6B35;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 24px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        {icon} {title}
    </div>
    """, unsafe_allow_html=True)


def render_welcome_header(user_name: str = None):
    """Render the main welcome header."""
    if user_name:
        greeting = f"Welcome back, {user_name.split()[0]}! 👋"
    else:
        greeting = "Welcome to NBA Live Stats Predictor! 🏀"
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 32px;">
        <h1 style="
            font-size: 2.5rem;
            background: linear-gradient(135deg, #FF6B35 0%, #FF8C5A 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        ">🏀 NBA Live Stats Predictor</h1>
        <p style="color: #9CA3AF; font-size: 1.1rem;">{greeting}</p>
    </div>
    """, unsafe_allow_html=True)


def render_login_page():
    """Render the login landing page for unauthenticated users."""
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h1 style="
            font-size: 3rem;
            background: linear-gradient(135deg, #FF6B35 0%, #FF8C5A 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 24px;
        ">🏀 NBA Live Stats Predictor</h1>
        <p style="color: #9CA3AF; font-size: 1.2rem; margin-bottom: 40px; max-width: 500px; margin-left: auto; margin-right: auto;">
            Get AI-powered predictions for NBA player performance using advanced Hidden Markov Models.
            Sign in to save your favorite players and teams!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #1F2937; border-radius: 12px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📈</div>
            <h3 style="color: #FF6B35; margin-bottom: 8px;">Live Data</h3>
            <p style="color: #9CA3AF; font-size: 0.9rem;">Real-time stats from NBA.com API</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #1F2937; border-radius: 12px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🤖</div>
            <h3 style="color: #FF6B35; margin-bottom: 8px;">AI Predictions</h3>
            <p style="color: #9CA3AF; font-size: 0.9rem;">HMM-based performance forecasting</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #1F2937; border-radius: 12px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">⭐</div>
            <h3 style="color: #FF6B35; margin-bottom: 8px;">Personalized</h3>
            <p style="color: #9CA3AF; font-size: 0.9rem;">Save favorites & track your players</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def render_quick_stats_row(stats: dict):
    """Render a row of quick stats metrics."""
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats.items()):
        with col:
            st.metric(label=label, value=value)
