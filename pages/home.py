import streamlit as st

st.set_page_config(page_title="ATP Tennis Feasibility Study", layout="wide")

st.title("ATP Tennis Performance Feasibility Study")

m1, m2 = st.columns(2)
m1.metric("Time Horizon", "2000–Present")
m2.metric("Code Tech", "Python Integration")

st.divider()

col1, col3, col4 = st.columns([2, 1, 1])

with col1:
    st.markdown("""
    <div style="
        background-color:#f8fafc;
        padding:24px;
        border-radius:16px;
        border:1px solid #e5e7eb;
    ">
        <h4 style="margin-top:0;">Introduction</h4>
        This project analyzes ATP tennis player performance using historical match data to better understand the patterns that influence match outcomes, with the goal of identifying trends that improve match and tournament win predictions.
        <br><br>
        To support this objective, an interactive dashboard has been developed to present these insights step by step, allowing users to explore key factors such as ranking differences, surface performance, recent form, and tournament context. The analysis concludes with a predictive model designed to estimate which player is more likely to win.
    </div>
    """, unsafe_allow_html=True)



with col3:
    st.markdown("""
    <div style="
        background-color:#ecfdf5;
        padding:20px;
        border-radius:16px;
        border:1px solid #a7f3d0;
        min-height:220px;
    ">
        <h4 style="margin-top:0;">Scope includes</h4>
        <ul>
            <li>Integration of datasets through common entities</li>
            <li>Systematic data cleaning</li>
            <li>Feature enrichment</li>
            <li>Descriptive analysis</li>
            <li>Light predictive modeling (baseline win probability)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="
        background-color:#fef2f2;
        padding:20px;
        border-radius:16px;
        border:1px solid #fecaca;
        min-height:220px;
    ">
        <h4 style="margin-top:0;">Out of scope</h4>
        <ul>
            <li>ELO model implementation</li>
            <li>Country-based tennis strength analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

colData, source2 = st.columns(2)

with colData:
    st.markdown(
        """
        <div style="
            background-color:#f8fafc;
            padding:20px;
            border-radius:16px;
            border:1px solid #e5e7eb;
            min-height:220px;
        ">
            <h4 style="margin-top:0;">Data Sources</h4>
            <b>Dataset A — ATP Tennis (2000–2026)</b><br>
            Historical match data from ATP tournaments spanning from 2000 to the present.
            <br><br>
            <b>Dataset B — Australian Open 2026</b><br>
            Tournament-specific dataset used to validate patterns identified in the historical data.
        </div>
        """,
        unsafe_allow_html=True
    )


with source2:
    st.markdown(
        """
        <div style="
            background-color:#f8fafc;
            padding:20px;
            border-radius:16px;
            border:1px solid #e5e7eb;
            min-height:220px;
        ">
            <h4 style="margin-top:0;">Integration Strategy</h4>
            <ul>
                <li>Normalize player names, dates, and rounds</li>
                <li>Build a robust player key</li>
                <li>Map surfaces, tournaments, and calendar context</li>
                <li>Compare Australian Open 2026 against historical ATP trends</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()

tab1, tab2, tab3 = st.columns(3)

with tab1:
    st.markdown(
        """
        <div style="
            background-color:#f8fafc;
            padding:20px;
            border-radius:16px;
            border:1px solid #e5e7eb;
            min-height:220px;
        ">
            <h4 style="margin-top:0;">Research Question 1</h4>
            How does the probability of winning vary based on both players' rankings and the difference between their rankings in ATP matches (2000–2026)?
        </div>
        """,
        unsafe_allow_html=True
    )

with tab2:
    st.markdown(
        """
        <div style="
            background-color:#f8fafc;
            padding:20px;
            border-radius:16px;
            border:1px solid #e5e7eb;
            min-height:220px;
        ">
            <h4 style="margin-top:0;">Research Question 2</h4>
            How does a player's performance change depending on the surface and the tournament context (Grand Slam vs. other tournaments)?
        </div>
        """,
        unsafe_allow_html=True
    )

with tab3:
    st.markdown(
        """
        <div style="
            background-color:#f8fafc;
            padding:20px;
            border-radius:16px;
            border:1px solid #e5e7eb;
            min-height:220px;
        ">
            <h4 style="margin-top:0;">Research Question 3</h4>
            To what extent can historical trends be used to simulate and predict the progression of players (quarterfinals to final) at the 2026 Australian Open?
        </div>
        """,
        unsafe_allow_html=True
    )
