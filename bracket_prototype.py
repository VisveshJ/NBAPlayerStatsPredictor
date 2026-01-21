import streamlit as st

def render_bracket():
    st.set_page_config(layout="wide")
    
    st.markdown("""
    <style>
    .bracket-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 20px;
        background-color: #161B22;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    .round {
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        height: 600px;
    }
    .matchup {
        display: flex;
        flex-direction: column;
        width: 180px;
        border: 1px solid #374151;
        border-radius: 6px;
        background: #1F2937;
        margin: 10px 0;
        position: relative;
    }
    .team {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        font-size: 0.9rem;
    }
    .team:first-child {
        border-bottom: 1px solid #374151;
    }
    .seed {
        font-size: 0.75rem;
        color: #9CA3AF;
        margin-right: 8px;
        width: 15px;
    }
    .team-name {
        font-weight: 500;
        flex-grow: 1;
    }
    .score {
        font-weight: bold;
        color: #10B981;
    }
    
    /* Connectors */
    .connector {
        position: absolute;
        right: -20px;
        top: 50%;
        width: 20px;
        height: 2px;
        background: #374151;
    }
    
    .bracket-title {
        text-align: center;
        color: #FF6B35;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='bracket-title'>NBA Playoff Bracket - Western Conference</div>", unsafe_allow_html=True)

    cols = st.columns([1, 0.2, 1, 0.2, 1, 1])
    
    with cols[0]:
        st.caption("First Round")
        st.markdown("""
        <div class='matchup'>
            <div class='team'><span class='seed'>1</span><span class='team-name'>Thunder</span><span class='score'>-</span></div>
            <div class='team'><span class='seed'>8</span><span class='team-name'>Warriors</span><span class='score'>-</span></div>
            <div class='connector'></div>
        </div>
        <div class='matchup'>
            <div class='team'><span class='seed'>4</span><span class='team-name'>Rockets</span><span class='score'>-</span></div>
            <div class='team'><span class='seed'>5</span><span class='team-name'>Timberwolves</span><span class='score'>-</span></div>
            <div class='connector'></div>
        </div>
        <div style='height: 40px;'></div>
        <div class='matchup'>
            <div class='team'><span class='seed'>2</span><span class='team-name'>Lakers</span><span class='score'>-</span></div>
            <div class='team'><span class='seed'>7</span><span class='team-name'>Clippers</span><span class='score'>-</span></div>
            <div class='connector'></div>
        </div>
        <div class='matchup'>
            <div class='team'><span class='seed'>3</span><span class='team-name'>Kings</span><span class='score'>-</span></div>
            <div class='team'><span class='seed'>6</span><span class='team-name'>Suns</span><span class='score'>-</span></div>
            <div class='connector'></div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.caption("Conf. Semifinals")
        st.markdown("""
        <div style='height: 50px;'></div>
        <div class='matchup'>
            <div class='team'><span class='seed'></span><span class='team-name'>TBD</span></div>
            <div class='team'><span class='seed'></span><span class='team-name'>TBD</span></div>
            <div class='connector'></div>
        </div>
        <div style='height: 150px;'></div>
        <div class='matchup'>
            <div class='team'><span class='seed'></span><span class='team-name'>TBD</span></div>
            <div class='team'><span class='seed'></span><span class='team-name'>TBD</span></div>
            <div class='connector'></div>
        </div>
        """, unsafe_allow_html=True)

    with cols[4]:
        st.caption("Conf. Finals")
        st.markdown("""
        <div style='height: 160px;'></div>
        <div class='matchup'>
            <div class='team'><span class='seed'></span><span class='team-name'>TBD</span></div>
            <div class='team'><span class='seed'></span><span class='team-name'>TBD</span></div>
            <div class='connector'></div>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[5]:
        st.caption("Finals")
        st.markdown("""
        <div style='height: 160px; display: flex; align-items: center; justify-content: center;'>
            <div style='font-size: 3rem;'>🏆</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_bracket()
