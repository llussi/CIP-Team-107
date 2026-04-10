import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard", page_icon="📊")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/atp_tennis.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()


# -------- Surface chart --------
def build_surface_chart(df: pd.DataFrame):

    surface_counts = df["Surface"].value_counts().reset_index()
    surface_counts.columns = ["Surface", "Matches"]

    fig = px.bar(
        surface_counts,
        x="Surface",
        y="Matches",
        title="Match Distribution by Surface",
        text="Matches",
    )

    fig.update_layout(template="plotly_white")

    return fig


# -------- Year chart --------
def build_year_chart(df: pd.DataFrame):

    df["Year"] = df["Date"].dt.year

    matches_per_year = df.groupby("Year").size().reset_index(name="Matches")

    df["Better_rank_winner"] = (
        ((df["Rank_1"] < df["Rank_2"]) & (df["Winner"] == df["Player_1"])) |
        ((df["Rank_2"] < df["Rank_1"]) & (df["Winner"] == df["Player_2"]))
    )

    better_rank_wins = df.groupby("Year")["Better_rank_winner"].sum().reset_index()
    better_rank_wins.rename(columns={"Better_rank_winner": "Better Rank Wins"}, inplace=True)

    data = matches_per_year.merge(better_rank_wins, on="Year")

    data_long = data.melt(
        id_vars="Year",
        value_vars=["Matches", "Better Rank Wins"],
        var_name="Type",
        value_name="Count"
    )

    fig = px.line(
        data_long,
        x="Year",
        y="Count",
        color="Type",
        markers=True,
        title="ATP Matches per Year vs Wins by Better Ranked Player",
    )

    fig.add_vline(x=2020, line_dash="dash", annotation_text="COVID-19")
    fig.update_layout(template="plotly_white")

    return fig


# -------- Top players --------
def build_top_players_chart(df: pd.DataFrame):

    top_players = df["Winner"].value_counts().reset_index()
    top_players.columns = ["Player", "Wins"]

    top10 = top_players.head(10)

    fig = px.bar(
        top10,
        x="Player",
        y="Wins",
        title="Top 10 ATP Players by Number of Wins",
        text="Wins"
    )

    fig.update_layout(
        xaxis_title="Player",
        yaxis_title="Number of Wins",
        template="plotly_white",
    )

    return fig


# -------- Metrics --------
def show_metrics(df: pd.DataFrame):

    total_matches = len(df)
    total_surfaces = df["Surface"].nunique()
    total_winners = df["Winner"].nunique()

    year_min = df["Date"].dt.year.min()
    year_max = df["Date"].dt.year.max()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Matches", f"{total_matches:,}")
    c2.metric("Surfaces", total_surfaces)
    c3.metric("Unique winners", total_winners)
    c4.metric("Time range", f"{year_min} - {year_max}")


# -------- Sidebar filters --------
surface_options = sorted(df["Surface"].dropna().unique())

selected_surfaces = st.sidebar.multiselect(
    "Surface",
    options=surface_options,
    default=surface_options,
)

filtered_df = df[df["Surface"].isin(selected_surfaces)]


# -------- Layout --------
st.title("Data Overview")

st.markdown("""
This section provides an overview of the dataset used in the analysis.  
It helps understand the structure, volume, and key characteristics of ATP match data before exploring deeper patterns and building predictive models.
""")

st.divider()

show_metrics(filtered_df)

st.markdown("""
<div style="
    background-color:#eff6ff;
    padding:16px;
    border-radius:12px;
    border:1px solid #bfdbfe;
    margin-bottom:20px;
">
    <b>Key idea:</b>  
    The dataset covers more than 20 years of ATP matches, allowing us to analyze how factors like surface, ranking, and time influence match outcomes.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(build_surface_chart(filtered_df), width="stretch")

with col2:
    st.plotly_chart(build_top_players_chart(filtered_df), width="stretch")

st.plotly_chart(build_year_chart(filtered_df), width="stretch")

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.markdown(
        """
        <div style="
            background-color:#f8fafc;
            padding:20px;
            border-radius:16px;
            border:1px solid #e5e7eb;
            min-height:220px;
        ">
            <h4 style="margin-top:0;">Key Takeaways</h4>
            <ul>
                <li>Hard court dominates ATP matches</li>
                <li>A small group of players accumulates a large share of total wins</li>
                <li>Match volume is stable over time, with a clear disruption during COVID-19</li>
                <li>Better-ranked players tend to win more consistently</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown("""
    <div style="
        background-color:#f8fafc;
        padding:18px;
        border-radius:12px;
        border:1px solid #e5e7eb;
    ">
    <h4 style="margin-top:0;"> What you'll explore next</h4>
    <p style="font-size:14px; line-height:1.6; margin-bottom:10px;">
    After understanding the dataset, the following sections dive deeper into the key factors that influence match outcomes:
    </p>
    
    <ul style="font-size:14px; line-height:1.6; margin-bottom:10px;">
        <li><b>Ranking impact on win probability</b> — analyze how ranking differences affect the likelihood of winning a match (RQ 1)</li>
        <li><b>Player comparison (H2H)</b> — compare two players across surfaces, form, and head-to-head performance (RQ 2)</li>
        <li><b>ATP Tournament Predictor</b> — simulate tournament outcomes using a predictive model (RQ 3)</li>
    </ul>
    
    </div>
    """, unsafe_allow_html=True)

