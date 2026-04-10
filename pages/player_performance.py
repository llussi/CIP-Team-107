import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Player Comparison", page_icon="🎾", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/atp_tennis.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Is_Grand_Slam"] = df["Series"].eq("Grand Slam")
    stats = pd.read_parquet("data/processed/stats_players.parquet")
    return df, stats

def get_all_players(df: pd.DataFrame):
    players = pd.unique(pd.concat([df["Player"]], ignore_index=True).dropna())
    return sorted(players.tolist())

def build_player_match_long(df: pd.DataFrame):
    p1 = df[["Date", "Year", "Surface", "Series", "Is_Grand_Slam", "Player_1", "Player_2", "Winner"]].copy()
    p1["Player"] = p1["Player_1"]
    p1["Opponent"] = p1["Player_2"]
    p1["Win"] = (p1["Winner"] == p1["Player_1"]).astype(int)

    p2 = df[["Date", "Year", "Surface", "Series", "Is_Grand_Slam", "Player_1", "Player_2", "Winner"]].copy()
    p2["Player"] = p2["Player_2"]
    p2["Opponent"] = p2["Player_1"]
    p2["Win"] = (p2["Winner"] == p2["Player_2"]).astype(int)

    long_df = pd.concat(
        [
            p1[["Date", "Year", "Surface", "Series", "Is_Grand_Slam", "Player", "Opponent", "Win"]],
            p2[["Date", "Year", "Surface", "Series", "Is_Grand_Slam", "Player", "Opponent", "Win"]],
        ],
        ignore_index=True,
    )

    return long_df.dropna(subset=["Player", "Opponent", "Surface", "Year"])

df, stats = load_data()
long_df = build_player_match_long(df)
all_players = get_all_players(long_df)



def get_player_summary(player_df: pd.DataFrame):
    matches = len(player_df)
    wins = int(player_df["Win"].sum())
    losses = matches - wins
    win_pct = (wins / matches * 100) if matches else 0

    surface_stats = (
        player_df.groupby("Surface", as_index=False)
        .agg(Matches=("Win", "count"), Wins=("Win", "sum"))
    )
    if not surface_stats.empty:
        surface_stats["Win_pct"] = surface_stats["Wins"] / surface_stats["Matches"] * 100
        best_surface = surface_stats.sort_values(["Win_pct", "Matches"], ascending=[False, False]).iloc[0]["Surface"]
    else:
        best_surface = "N/A"

    slam_df = player_df[player_df["Is_Grand_Slam"]]
    slam_win_pct = (slam_df["Win"].mean() * 100) if not slam_df.empty else 0

    return {
        "matches": matches,
        "wins": wins,
        "losses": losses,
        "win_pct": win_pct,
        "best_surface": best_surface,
        "slam_win_pct": slam_win_pct,
    }

def render_player_card(player_name: str, player_df: pd.DataFrame):
    stats = get_player_summary(player_df)

    st.markdown(
        f"""
        <div style="
            border:1px solid #e6e6e6;
            border-radius:16px;
            padding:18px;
            background-color:#fafafa;
            min-height:220px;
        ">
            <h3 style="margin-top:0;">{player_name}</h3>
            <p><b>Matches:</b> {stats["matches"]:,}</p>
            <p><b>Wins:</b> {stats["wins"]:,}</p>
            <p><b>Losses:</b> {stats["losses"]:,}</p>
            <p><b>Win rate:</b> {stats["win_pct"]:.1f}%</p>
            <p><b>Best surface:</b> {stats["best_surface"]}</p>
            <p><b>Grand Slam win rate:</b> {stats["slam_win_pct"]:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_surface_seasons_chart(long_df: pd.DataFrame, selected_players: list[str]):
    stats = (
        long_df[long_df["Player"].isin(selected_players)]
        .groupby(["Player", "Year", "Surface"], as_index=False)
        .agg(Matches=("Win", "count"), Wins=("Win", "sum"))
    )

    stats = stats[stats["Matches"] >= 3].copy()
    stats["Win_pct"] = stats["Wins"] / stats["Matches"] * 100

    fig = px.line(
        stats,
        x="Year",
        y="Win_pct",
        color="Surface",
        facet_col="Player",
        facet_col_wrap=2,
        markers=True,
        hover_data=["Matches", "Wins"],
        title="Win percentage by surface across seasons",
        labels={
            "Year": "Season",
            "Win_pct": "Win percentage (%)",
            "Surface": "Surface",
        },
        color_discrete_map={
            "Grass": "#2EA23C",
            "Hard": "#2472BC",
            "Clay": "#DA6A2F",
        }
    )

    fig.update_layout(template="plotly_white", height=500)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig

def build_tournament_context_chart(long_df: pd.DataFrame, selected_players: list[str]):
    data = long_df[long_df["Player"].isin(selected_players)].copy()

    stats = (
        data.groupby(["Player", "Series"], as_index=False)
        .agg(Matches=("Win", "count"), Wins=("Win", "sum"))
    )
    stats["Win_pct"] = stats["Wins"] / stats["Matches"] * 100

    fig = px.bar(
        stats,
        x="Win_pct",
        y="Series",
        color="Player",
        barmode="group",
        orientation="h",
        text_auto=".1f",
        title="Win percentage by tournament context",
        hover_data=["Matches", "Wins"],
        labels={
            "Series": "Series",
            "Win_pct": "Win percentage (%)",
            "Player": "Player",
        },
        color_discrete_map={
            selected_players[0]: "#1f77b4",
            selected_players[1]: "#d62728",
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        yaxis=dict(categoryorder="total ascending")
    )

    return fig

def build_h2h_summary(df: pd.DataFrame, player_a: str, player_b: str):
    h2h = df[
        (
            ((df["Player_1"] == player_a) & (df["Player_2"] == player_b))
            | ((df["Player_1"] == player_b) & (df["Player_2"] == player_a))
        )
    ].copy()

    total_matches = len(h2h)
    wins_a = int((h2h["Winner"] == player_a).sum())
    wins_b = int((h2h["Winner"] == player_b).sum())

    by_surface = (
        h2h.groupby(["Surface", "Winner"], as_index=False)
        .size()
        .rename(columns={"size": "Matches"})
    )

    return h2h, total_matches, wins_a, wins_b, by_surface


def build_h2h_chart(by_surface: pd.DataFrame, player_a: str, player_b: str):
    if by_surface.empty:
        return None

    df = by_surface.copy()

    df["Total_surface_matches"] = df.groupby("Surface")["Matches"].transform("sum")
    df["Win_pct"] = df["Matches"] / df["Total_surface_matches"] * 100

    fig = px.bar(
        df,
        x="Win_pct",
        y="Surface",
        color="Winner",
        orientation="h",
        barmode="stack",
        custom_data=["Matches"],
        title=f"H2H win percentage by surface: {player_a} vs {player_b}",
        labels={
            "Win_pct": "Wins",
            "Surface": "Surface",
            "Winner": "Player",
        },
        category_orders={
            "Winner": [player_a, player_b],
            "Surface": ["Clay", "Hard", "Grass"],
        },
        color_discrete_map={
            player_a: "#1f77b4",
            player_b: "#d62728",
        },
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Surface: %{y}<br>"
            "Wins: %{customdata[0]}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_range=[0, 100],
    )

    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title=""
    )

    return fig


st.title("🎾 Player Comparison")
st.markdown("Compare two ATP players across surfaces, tournament context, and head-to-head results.")

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    player_a = st.selectbox("Player 1", options=all_players, index=all_players.index("Djokovic N.") if "Djokovic N." in all_players else 0)

with col_sel2:
    default_index_b = all_players.index("Nadal R.") if "Nadal R." in all_players else min(1, len(all_players) - 1)
    player_b = st.selectbox("Player 2", options=all_players, index=default_index_b)

if player_a == player_b:
    st.warning("Select two different players.")
    st.stop()

player_a_df = long_df[long_df["Player"] == player_a].copy()
player_b_df = long_df[long_df["Player"] == player_b].copy()

st.subheader("Player cards")

card1, card2 = st.columns(2)
with card1:
    render_player_card(player_a, player_a_df)

with card2:
    render_player_card(player_b, player_b_df)


st.subheader("Head-to-head")

h2h_df, total_h2h, wins_a, wins_b, by_surface = build_h2h_summary(df, player_a, player_b)

m2, m1, m3 = st.columns([1,1,1])
m1.metric("Total H2H matches", total_h2h)
m2.metric(player_a, wins_a)
m3.metric(player_b, wins_b)


st.plotly_chart(
    build_surface_seasons_chart(long_df, [player_a, player_b]),
    width='stretch',
)

col1, col2 = st.columns(2)

with col1:
    h2h_fig = build_h2h_chart(by_surface, player_a, player_b)
    if h2h_fig is not None:
        st.plotly_chart(h2h_fig, width='stretch')
    else:
        st.info("There are no H2H matches between these two players in the dataset")

with col2:
    st.plotly_chart(
        build_tournament_context_chart(long_df, [player_a, player_b]),
        width='stretch',
    )


st.subheader("Head-to-head Matches")




with st.expander("View H2H matches"):
    if h2h_df.empty:
        st.write("There is no H2H data between these two players.")
    else:
        st.dataframe(
            h2h_df[["Date", "Surface", "Series", "Player_1", "Player_2", "Winner"]]
            .sort_values("Date", ascending=False),
            width='stretch',
        )

