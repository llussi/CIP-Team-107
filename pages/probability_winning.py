import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Probability Winning", page_icon="📊")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/atp_tennis.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

def build_diff_ranking(df):

    ranking_df = df.copy()

    # Create ranking-difference groups
    bins = [0, 10, 20, 30, 50, 80, 120, 1000]
    labels = [
        "1-10",
        "11-20",
        "21-30",
        "31-50",
        "51-80",
        "81-120",
        "120+"
    ]

    ranking_df["Rank_diff_group"] = pd.cut(
        ranking_df["Rank_diff"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Probability that the better-ranked player wins
    rank_effect = (
        ranking_df
        .groupby("Rank_diff_group")["Better_rank_winner"]
        .mean()
        .reset_index()
    )

    rank_effect["Win_probability"] = rank_effect["Better_rank_winner"] * 100

    # Chart
    fig = px.line(
        rank_effect,
        x="Rank_diff_group",
        y="Win_probability",
        markers=True,
        title="Probability that the higher-ranked player wins by ranking difference",
        labels={
            "Rank_diff_group": "Ranking difference",
            "Win_probability": "Win probability (%)"
        }
    )

    fig.update_layout(template="plotly_white")

    return fig

def build_win_prob_by_player_rank(df):

    data = df.copy()

    # Identify the better-ranked player (lower number)
    data["Better_player_rank"] = data[["Rank_1", "Rank_2"]].min(axis=1)

    # Check whether the better-ranked player wins
    data["Better_rank_wins"] = (
        ((data["Rank_1"] < data["Rank_2"]) & (data["Winner"] == data["Player_1"])) |
        ((data["Rank_2"] < data["Rank_1"]) & (data["Winner"] == data["Player_2"]))
    )

    # Create player ranking bins
    bins = [0, 5, 10, 20, 50, 100, 200, 500]
    labels = [
        "1-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        "101-200",
        "200+"
    ]

    data["Rank_group"] = pd.cut(
        data["Better_player_rank"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Calculate probability
    rank_effect = (
        data
        .groupby("Rank_group")["Better_rank_winner"]
        .mean()
        .reset_index()
    )

    rank_effect["Win_probability"] = rank_effect["Better_rank_winner"] * 100

    # Chart
    fig = px.line(
        rank_effect,
        x="Rank_group",
        y="Win_probability",
        markers=True,
        title="Win probability of higher-ranked player by ranking level",
        labels={
            "Rank_group": "Player ranking",
            "Win_probability": "Win probability (%)"
        }
    )

    fig.update_layout(template="plotly_white")
    fig.update_traces(line_width=3)
    fig.update_yaxes(range=[50, 100])

    return fig


def build_heatmap(df, surface=None, series=None):
    data = df.copy()

    # Player 1 perspective
    p1 = data[["Rank_1", "Rank_2", "Player_1", "Player_2", "Winner", "Surface", "Series"]].copy()
    p1["Player_rank"] = p1["Rank_1"]
    p1["Opponent_rank"] = p1["Rank_2"]
    p1["Win"] = (p1["Winner"] == p1["Player_1"]).astype(int)

    # Player 2 perspective
    p2 = data[["Rank_1", "Rank_2", "Player_1", "Player_2", "Winner", "Surface", "Series"]].copy()
    p2["Player_rank"] = p2["Rank_2"]
    p2["Opponent_rank"] = p2["Rank_1"]
    p2["Win"] = (p2["Winner"] == p2["Player_2"]).astype(int)

    # Join
    data = pd.concat([p1, p2], ignore_index=True)

    # Filter by surface
    if surface is not None:
        if isinstance(surface, (list, tuple, set)):
            if len(surface) > 0:
                data = data[data["Surface"].isin(surface)]
        else:
            data = data[data["Surface"] == surface]

    # Filter by series
    if series is not None:
        if isinstance(series, (list, tuple, set)):
            if len(series) > 0:
                data = data[data["Series"].isin(series)]
        else:
            data = data[data["Series"] == series]

    bins = [0, 5, 10, 20, 50, 100, 200, 500, 9999]
    labels = ["1-5", "6-10", "11-20", "21-50", "51-100", "101-200", "200-500", "500+"]

    data["Player_group"] = pd.cut(
        data["Player_rank"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    data["Opponent_group"] = pd.cut(
        data["Opponent_rank"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    heatmap = (
        data.groupby(["Player_group", "Opponent_group"], observed=True)["Win"]
        .mean()
        .reset_index()
    )

    heatmap["Win_prob"] = heatmap["Win"] * 100

    pivot = heatmap.pivot(
        index="Player_group",
        columns="Opponent_group",
        values="Win_prob"
    )

    title_parts = ["Win probability heatmap"]
    if surface:
        title_parts.append(f"Surface: {surface}")
    if series:
        title_parts.append(f"Series: {series}")

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title=" | ".join(title_parts)
    )

    fig.update_layout(
        xaxis_title="Opponent ranking",
        yaxis_title="Player ranking"
    )

    return fig

# =========================
# SECTION 1: Ranking analysis
# =========================

st.title("Ranking impact on win probability")

st.markdown("""

This section explores how player rankings influence match results.  
We analyze whether higher-ranked players consistently outperform lower-ranked opponents and how ranking differences translate into win probability.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        build_diff_ranking(df),
        width='stretch'
    )

with col2:
    st.plotly_chart(
        build_win_prob_by_player_rank(df),
        width='stretch'
    )

st.markdown("""
<div style="
    background-color:#eff6ff;
    padding:16px;
    border-radius:12px;
    border:1px solid #bfdbfe;
    margin-bottom:20px;
">
    <b>Key idea: </b>  
     The larger the ranking gap, the higher the probability that the better-ranked player wins. Top-ranked players (1–10) show significantly higher win rates compared to lower-ranked groups.
</div>
""", unsafe_allow_html=True)


# =========================
# SECTION 2: Heatmap + filters
# =========================

st.subheader("Win probability heatmap (ranking vs ranking)")

col_filters, col_chart = st.columns([1, 3])  # sidebar-like layout

with col_filters:

    st.markdown("### Filters")

    surface_selected = st.selectbox(
        "Surface",
        options=["All"] + sorted(df["Surface"].dropna().unique().tolist())
    )

    series_selected = st.selectbox(
        "Series",
        options=["All"] + sorted(df["Series"].dropna().unique().tolist())
    )

    surface_filter = None if surface_selected == "All" else surface_selected
    series_filter = None if series_selected == "All" else series_selected


with col_chart:

    fig = build_heatmap(df, surface=surface_filter, series=series_filter)

    st.plotly_chart(fig, width='stretch')