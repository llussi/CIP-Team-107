from operator import index

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="ATP Tournament Bracket", page_icon="🎾", layout="wide")

def load_data():
    df = pd.read_parquet("data/processed/atp_tennis.parquet")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/atp_match_model.pkl")


@st.cache_data
def build_player_long_df(df: pd.DataFrame):
    p1 = df[["Date", "Surface", "Series", "Round", "Player_1", "Player_2", "Winner", "Rank_1", "Rank_2"]].copy()
    p1["Player"] = p1["Player_1"]
    p1["Opponent"] = p1["Player_2"]
    p1["Player_rank"] = p1["Rank_1"]
    p1["Opponent_rank"] = p1["Rank_2"]
    p1["Win"] = (p1["Winner"] == p1["Player_1"]).astype(int)

    p2 = df[["Date", "Surface", "Series", "Round", "Player_1", "Player_2", "Winner", "Rank_1", "Rank_2"]].copy()
    p2["Player"] = p2["Player_2"]
    p2["Opponent"] = p2["Player_1"]
    p2["Player_rank"] = p2["Rank_2"]
    p2["Opponent_rank"] = p2["Rank_1"]
    p2["Win"] = (p2["Winner"] == p2["Player_2"]).astype(int)

    data = pd.concat([p1, p2], ignore_index=True)
    data["Rank_diff"] = data["Opponent_rank"] - data["Player_rank"]

    return data

model = load_model()

def predict_match(player_features):

    prob = model.predict_proba(player_features)[0][1]

    return prob * 100

def compute_last5_form(data: pd.DataFrame):
    data = data.sort_values("Date").copy()
    data["Last5_win_rate"] = (
        data.groupby("Player")["Win"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )
    latest = data.groupby("Player").tail(1)
    return latest[["Player", "Last5_win_rate"]]

def compute_surface_win_rate(data: pd.DataFrame):
    stats = (
        data.groupby(["Player", "Surface"], as_index=False)
        .agg(Matches=("Win", "count"), Wins=("Win", "sum"))
    )
    stats["Surface_win_rate"] = stats["Wins"] / stats["Matches"]
    return stats[["Player", "Surface", "Surface_win_rate"]]

def compute_overall_win_rate(data: pd.DataFrame):
    stats = (
        data.groupby("Player", as_index=False)
        .agg(Matches=("Win", "count"), Wins=("Win", "sum"))
    )
    stats["Overall_win_rate"] = stats["Wins"] / stats["Matches"]
    return stats[["Player", "Overall_win_rate"]]

def compute_vs_better_rate(data: pd.DataFrame):
    temp = data.copy()
    temp["Better_opp"] = temp["Opponent_rank"] < temp["Player_rank"]

    stats = (
        temp[temp["Better_opp"]]
        .groupby("Player", as_index=False)
        .agg(Matches=("Win", "count"), Wins=("Win", "sum"))
    )

    stats["Win_vs_better"] = stats["Wins"] / stats["Matches"]
    return stats[["Player", "Win_vs_better"]]

def compute_vs_worse_rate(data: pd.DataFrame):
    temp = data.copy()
    temp["Worse_opp"] = temp["Opponent_rank"] > temp["Player_rank"]

    stats = (
        temp[temp["Worse_opp"]]
        .groupby("Player", as_index=False)
        .agg(Matches=("Win", "count"), Wins=("Win", "sum"))
    )

    stats["Win_vs_worse"] = stats["Wins"] / stats["Matches"]
    return stats[["Player", "Win_vs_worse"]]

df = load_data()
long_df = build_player_long_df(df)
players = sorted(long_df["Player"].dropna().unique().tolist())
surfaces = sorted(df["Surface"].dropna().unique().tolist())
series_options = sorted(df["Series"].dropna().unique().tolist())
round_options = sorted(df["Round"].dropna().unique().tolist())

latest_form_df = compute_last5_form(long_df)
latest_form_map = dict(zip(latest_form_df["Player"], latest_form_df["Last5_win_rate"]))

surface_wr_df = compute_surface_win_rate(long_df)
surface_wr_map = {
    (row["Player"], row["Surface"]): row["Surface_win_rate"]
    for _, row in surface_wr_df.iterrows()
}

overall_wr_df = compute_overall_win_rate(long_df)
overall_wr_map = dict(zip(overall_wr_df["Player"], overall_wr_df["Overall_win_rate"]))

vs_better_df = compute_vs_better_rate(long_df)
vs_better_map = dict(zip(vs_better_df["Player"], vs_better_df["Win_vs_better"]))

vs_worse_df = compute_vs_worse_rate(long_df)
vs_worse_map = dict(zip(vs_worse_df["Player"], vs_worse_df["Win_vs_worse"]))

latest_ranks = (
    long_df.sort_values("Date")
    .groupby("Player")
    .tail(1)[["Player", "Player_rank"]]
)
rank_map = dict(zip(latest_ranks["Player"], latest_ranks["Player_rank"]))

player_labels = {
    player: f"{player} (#{int(rank_map[player])})"
    if pd.notna(rank_map.get(player))
    else player
    for player in players
}


def add_vertical_space(lines=1):
    for _ in range(lines):
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

def predict_match_probability(
    player1,
    player2,
    tournament:list[str]
):
    predict_game = pd.DataFrame({
        "Round_num": [5],
        "Win_rate": [overall_wr_map.get(player1)],
        "Surface_win_rate": [surface_wr_map.get((player1, surface))],
        "Win_vs_better": [vs_better_map.get(player1)],
        "Rank_diff": [rank_map.get(player1) - rank_map.get(player2)],
    })
    predict_game_player2 = pd.DataFrame({
        "Round_num": [5],
        "Win_rate": [overall_wr_map.get(player2)],
        "Surface_win_rate": [surface_wr_map.get((player2, surface))],
        "Win_vs_better": [vs_better_map.get(player2)],
        "Rank_diff": [rank_map.get(player2) - rank_map.get(player1)],
    })
    win_pred_player_1 = predict_match(predict_game)
    win_pred_player_2 = predict_match(predict_game_player2)

    total = win_pred_player_1 + win_pred_player_2
    if total == 0:
        return 0, 0
    return (win_pred_player_1 / total) * 100, (win_pred_player_2 / total) * 100


def match_selector(
    list_players_1,
    list_players_2,
    match_key,
    tournament:list[str]
):
    # -----------------------------
    # SELECT PLAYERS
    # -----------------------------

    list_players_1 = list_players_1 or []
    list_players_2 = list_players_2 or []

    key1 = f"{match_key}_player1"
    key2 = f"{match_key}_player2"

    if len(list_players_1) == 2:
        if key1 not in st.session_state or st.session_state[key1] not in list_players_1:
            st.session_state[key1] = list_players_1[0]

    if len(list_players_2) == 2:
        if key2 not in st.session_state or st.session_state[key2] not in list_players_2:
            st.session_state[key2] = list_players_2[0]

    player1 = st.selectbox(
        "Player 1",
        options=list_players_1,
        index=None,
        placeholder="Choose an option",
        key=f"{match_key}_player1",
        label_visibility="collapsed"
    )

    player2 = st.selectbox(
        "Player 2",
        options=list_players_2,
        index=None,
        placeholder="Choose an option",
        key=f"{match_key}_player2",
        label_visibility="collapsed"
    )

    if player1 is None or player2 is None:
        return None

    if player1 == player2:
        st.warning("Players must be different")
        return None

    # Reset state if the match changes
    match_id = f"{player1}_{player2}"
    state_key = f"{match_key}_match"

    if st.session_state.get(state_key) != match_id:
        st.session_state.pop(match_key, None)
        st.session_state[state_key] = match_id

    prob_a, prob_b = predict_match_probability(
        player1,
        player2,
        tournament
    )
    st.markdown(
        f"""
        <div style="font-size:11px; margin-bottom:1px;">
            {player1}: {prob_a:.1f}% | {player2}: {prob_b:.1f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    if prob_a > prob_b:
        return [player1, player2]
    else:
        return [player2, player1]




def render_bracket(players_list, tournament:list[str]):
    tournament_rounds = {
        "Grand Slam": ["The Final", "Semifinals", "Quarterfinals", "4th Round", "3rd Round", "2nd Round", "1st Round"],
        "Masters 1000": ["The Final", "Semifinals", "Quarterfinals", "3rd Round", "2nd Round", "1st Round",None],
        "ATP500": ["The Final", "Semifinals", "Quarterfinals", "2nd Round", "1st Round",None,None],
        "ATP250": ["The Final", "Semifinals", "Quarterfinals", "2nd Round", "1st Round",None,None],
    }
    type = tournament[0]
    stage = tournament[2]

    if tournament_rounds[type][6] and tournament_rounds[type][6] == stage :
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,1,1,1,1])

        with col1:
            st.markdown("## " + tournament_rounds[type][6], unsafe_allow_html=True)

            r128 = []
            for i in range(1, 65):
                match = match_selector(players_list, players_list, f"r128_{i}", tournament)
                r128.append(match)
                if i < 64:
                    add_vertical_space(1)

        with col2:
            st.markdown("## " + tournament_rounds[type][5], unsafe_allow_html=True)

            add_vertical_space(4)

            r64 = []
            for i in range(0, 64, 2):
                match = match_selector(r128[i], r128[i + 1], f"r64_{i // 2 + 1}", tournament)
                r64.append(match)

                if i < 62:
                    add_vertical_space(9)

        with col3:
            st.markdown("## " + tournament_rounds[type][4], unsafe_allow_html=True)

            add_vertical_space(12)

            r32 = []
            for i in range(0, 32, 2):
                match = match_selector(r64[i], r64[i + 1], f"r32_{i // 2 + 1}", tournament)
                r32.append(match)

                if i < 30:
                    add_vertical_space(25)

        with col4:
            st.markdown("## " + tournament_rounds[type][3], unsafe_allow_html=True)

            add_vertical_space(28)

            r16 = []
            for i in range(0, 16, 2):
                match = match_selector(r32[i], r32[i + 1], f"r16_{i // 2 + 1}", tournament)
                r16.append(match)

                if i < 14:
                    add_vertical_space(57)

        with col5:
            st.markdown("## " + tournament_rounds[type][2], unsafe_allow_html=True)

            add_vertical_space(60)

            qf = []
            for i in range(0, 8, 2):
                match = match_selector(r16[i], r16[i + 1], f"qf_{i // 2 + 1}", tournament)
                qf.append(match)

                if i < 6:
                    add_vertical_space(120)

        with col6:
            st.markdown("## Semifinals")

            add_vertical_space(60)
            sf1 = match_selector(qf[0], qf[1], "sf1", tournament)

            add_vertical_space(120)
            sf2 = match_selector(qf[2], qf[3], "sf2", tournament)

        with col7:
            st.markdown("## Final")

            add_vertical_space(123)
            final = match_selector(sf1, sf2, "final", tournament)



    if tournament_rounds[type][5] == stage :
        col1, col2, col3, col4, col5, col6 = st.columns([1.2,1.2,1.2,1.2, 1.2, 1])

        with col1:
            st.markdown("## " + tournament_rounds[type][5], unsafe_allow_html=True)

            r64 = []
            for i in range(0, 64, 2):
                match = match_selector(players_list, players_list, f"r64_{i // 2 + 1}", tournament)
                r64.append(match)

                if i < 62:
                    add_vertical_space(1)

        with col2:
            st.markdown("## " + tournament_rounds[type][4], unsafe_allow_html=True)

            add_vertical_space(4)

            r32 = []
            for i in range(0, 32, 2):
                match = match_selector(r64[i], r64[i + 1], f"r32_{i // 2 + 1}", tournament)
                r32.append(match)

                if i < 30:
                    add_vertical_space(9)

        with col3:
            st.markdown("## " + tournament_rounds[type][3], unsafe_allow_html=True)

            add_vertical_space(12)

            r16 = []
            for i in range(0, 16, 2):
                match = match_selector(r32[i], r32[i + 1], f"r16_{i // 2 + 1}", tournament)
                r16.append(match)

                if i < 14:
                    add_vertical_space(25)

        with col4:
            st.markdown("## " + tournament_rounds[type][2], unsafe_allow_html=True)

            add_vertical_space(28)

            qf = []
            for i in range(0, 8, 2):
                match = match_selector(r16[i], r16[i + 1], f"qf_{i // 2 + 1}", tournament)
                qf.append(match)

                if i < 6:
                    add_vertical_space(57)

        with col5:
            st.markdown("## Semifinals")

            add_vertical_space(60)
            sf1 = match_selector(qf[0], qf[1], "sf1", tournament)

            add_vertical_space(120)
            sf2 = match_selector(qf[2], qf[3], "sf2", tournament)

        with col6:
            st.markdown("## Final")

            add_vertical_space(123)
            final = match_selector(sf1, sf2, "final", tournament)


    if tournament_rounds[type][4] == stage :
        col1, col2, col3, col4, col5 = st.columns([1.2,1.2,1.2, 1.2, 1])

        with col1:
            st.markdown("## " + tournament_rounds[type][4], unsafe_allow_html=True)

            r32 = []
            for i in range(0, 32, 2):
                match = match_selector(players_list, players_list, f"r32_{i // 2 + 1}", tournament)
                r32.append(match)

                if i < 30:
                    add_vertical_space(1)

        with col2:
            st.markdown("## " + tournament_rounds[type][3], unsafe_allow_html=True)

            add_vertical_space(4)

            r16 = []
            for i in range(0, 16, 2):
                match = match_selector(r32[i], r32[i + 1], f"r16_{i // 2 + 1}", tournament)
                r16.append(match)

                if i < 14:
                    add_vertical_space(9)

        with col3:
            st.markdown("## " + tournament_rounds[type][2], unsafe_allow_html=True)

            add_vertical_space(12)

            qf = []
            for i in range(0, 8, 2):
                match = match_selector(r16[i], r16[i + 1], f"qf_{i // 2 + 1}", tournament)
                qf.append(match)

                if i < 6:
                    add_vertical_space(25)

        with col4:
            st.markdown("## Semifinals")

            add_vertical_space(28)
            sf1 = match_selector(qf[0], qf[1], "sf1", tournament)

            add_vertical_space(57)
            sf2 = match_selector(qf[2], qf[3], "sf2", tournament)

        with col5:
            st.markdown("## Final")

            add_vertical_space(60)
            final = match_selector(sf1, sf2, "final", tournament)


    elif tournament_rounds[type][3] == stage :
        col1, col2, col3, col4 = st.columns([1.2,1.2, 1.2, 1])

        with col1:
            st.markdown("## " + tournament_rounds[type][3], unsafe_allow_html=True)

            r16 = []
            for i in range(0, 16, 2):
                match = match_selector(players_list, players_list, f"r16_{i // 2 + 1}", tournament)
                r16.append(match)

                if i < 14:
                    add_vertical_space(1)

        with col2:
            st.markdown("## " + tournament_rounds[type][2], unsafe_allow_html=True)

            add_vertical_space(4)

            qf = []
            for i in range(0, 8, 2):
                match = match_selector(r16[i], r16[i + 1], f"qf_{i // 2 + 1}", tournament)
                qf.append(match)

                if i < 6:
                    add_vertical_space(9)

        with col3:
            st.markdown("## Semifinals")

            add_vertical_space(12)
            sf1 = match_selector(qf[0], qf[1], "sf1", tournament)

            add_vertical_space(25)
            sf2 = match_selector(qf[2], qf[3], "sf2", tournament)

        with col4:
            st.markdown("## Final")

            add_vertical_space(28)
            final = match_selector(sf1, sf2, "final", tournament)


    elif tournament_rounds[type][2] == stage: # Quarterfinals
        col1, col2, col3 = st.columns([1.2, 1.2, 1])

        with col1:
            st.markdown("## " + tournament_rounds[type][2], unsafe_allow_html=True)

            qf = []
            for i in range(0, 8, 2):
                match = match_selector(players_list, players_list, f"qf_{i // 2 + 1}", tournament)
                qf.append(match)

                if i < 6:
                    add_vertical_space(1)


        with col2:
            st.markdown("## Semifinals")

            add_vertical_space(4)
            sf1 = match_selector(qf[0], qf[1], "sf1", tournament)

            add_vertical_space(9)
            sf2 = match_selector(qf[2], qf[3], "sf2", tournament)

        with col3:
            st.markdown("## Final")

            add_vertical_space(12)
            final = match_selector(sf1, sf2, "final", tournament)

    elif tournament_rounds[type][1] == stage: # Semifinals
        col2, col3 = st.columns([1.2, 1])

        with col2:
            st.markdown("## Semifinals")

            add_vertical_space(2)
            sf1 = match_selector(players_list, players_list, "sf1", tournament)

            add_vertical_space(5)
            sf2 = match_selector(players_list, players_list, "sf2", tournament)

        with col3:
            st.markdown("## Final")

            add_vertical_space(8)
            final = match_selector(sf1, sf2, "final", tournament)

    else:
        col3 = st.columns([1])[0]

        sf1 = players_list
        sf2 = players_list

        with col3:
            st.markdown("## Final")

            add_vertical_space(6)
            final = match_selector(sf1, sf2, "final", tournament)

    st.markdown("---")
    if final:
        st.success(f"🏆 Champion: {final[0]}")


TOURNAMENT_ROUNDS = {
    "Grand Slam": ["1st Round", "2nd Round", "3rd Round", "4th Round", "Quarterfinals", "Semifinals", "The Final"],
    "Masters 1000": ["1st Round", "2nd Round", "3rd Round", "Quarterfinals", "Semifinals", "The Final"],
    "ATP500": ["1st Round", "2nd Round", "Quarterfinals", "Semifinals", "The Final"],
    "ATP250": ["1st Round", "2nd Round", "Quarterfinals", "Semifinals", "The Final"],
}

st.title("ATP Match Predictor")
st.markdown("Win probability prediction using ranking, recent form, surface, and tournament context")

st.subheader("Model summary")

m1, m2, m3 = st.columns(3)
m1.metric("Algorithm", "Loaded model")
m2.metric("Players", f"{long_df['Player'].nunique():,}")
m3.metric("Matches", f"{len(long_df):,}")

st.markdown("### Tournament filters")

f1, f2, f3 = st.columns(3)

series = f1.selectbox(
    "Tournament type",
    options=["Grand Slam", "Masters 1000", "ATP500", "ATP250"],
    index=0
)

surface = f2.selectbox(
    "Surface",
    options=surfaces
)


round_name = f3.selectbox(
    "Round to simulate",
    options=TOURNAMENT_ROUNDS[series],
    index=4
)

tourney = [series, surface, round_name]

render_bracket(players, tourney)