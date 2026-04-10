import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import roc_auc_score, accuracy_score

st.set_page_config(page_title="ATP Match Predictor", page_icon="🎾", layout="wide")


@st.cache_data
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


def build_feature_row(player, opponent, surface, series, round_name, latest_form, surface_wr, overall_wr, vs_better, ranks):
    player_rank = ranks.get(player)
    opponent_rank = ranks.get(opponent)

    row = pd.DataFrame({
        "Surface": [surface],
        "Series": [series],
        "Round": [round_name],
        "Player_rank": [player_rank],
        "Opponent_rank": [opponent_rank],
        "Rank_diff": [opponent_rank - player_rank if pd.notna(player_rank) and pd.notna(opponent_rank) else None],
    })

    row = row.merge(latest_form, how="left", left_on="Player_rank", right_on="Player_rank") if False else row

    row["Last5_win_rate"] = latest_form.get(player)
    row["Overall_win_rate"] = overall_wr.get(player)
    row["Win_vs_better"] = vs_better.get(player)
    row["Surface_win_rate"] = surface_wr.get((player, surface))

    return row


def build_calibration_plot(eval_df: pd.DataFrame):
    temp = eval_df.copy()
    temp["prob_bin"] = pd.cut(temp["Predicted_prob"], bins=10)
    cal = temp.groupby("prob_bin", as_index=False).agg(
        Predicted=("Predicted_prob", "mean"),
        Actual=("Actual", "mean"),
        Count=("Actual", "count")
    )

    fig = px.line(
        cal,
        x="Predicted",
        y="Actual",
        markers=True,
        hover_data=["Count"],
        title="Calibration plot",
        labels={"Predicted": "Predicted probability", "Actual": "Observed win rate"}
    )
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash")
    )
    fig.update_layout(template="plotly_white")
    return fig


df = load_data()
model = load_model()
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

st.title("ATP Match Predictor")
st.markdown("Predicción de probabilidad de victoria usando ranking, forma reciente, superficie y contexto del torneo.")

st.subheader("Model summary")

m1, m2, m3 = st.columns(3)
m1.metric("Algorithm", "Loaded model")
m2.metric("Players", f"{long_df['Player'].nunique():,}")
m3.metric("Matches", f"{len(long_df):,}")

st.markdown("---")

st1, st2, st3 = st.columns(3)

surface = st1.selectbox("Surface", options=surfaces)
series = st2.selectbox("Series", options=series_options)
round_name = st3.selectbox("Round", options=round_options)



left, right = st.columns([1, 2])


with left:
    st.markdown("### Match inputs")

    player_1 = st.selectbox(
        "Player 1",
        options=players,
        format_func=lambda x: player_labels[x]
    )
    player_2 = st.selectbox(
        "Player 2",
        options=players,
        index=1 if len(players) > 1 else 0,
        format_func=lambda x: player_labels[x]
    )

    if player_1 == player_2:
        st.warning("Selecciona dos jugadores distintos.")
        st.stop()

    feature_row_p1 = build_feature_row(
        player_1, player_2, surface, series, round_name,
        latest_form_map, surface_wr_map, overall_wr_map, vs_better_map, rank_map
    )

    feature_row_p2 = build_feature_row(
        player_2, player_1, surface, series, round_name,
        latest_form_map, surface_wr_map, overall_wr_map, vs_better_map, rank_map
    )

    st.markdown("### Engineered features")
    st.dataframe(feature_row_p1, use_container_width=True)

with right:
    st.markdown("### Prediction")

    prob_p1 = model.predict_proba(feature_row_p1)[0][1]
    prob_p2 = model.predict_proba(feature_row_p2)[0][1]

    total = prob_p1 + prob_p2
    if total > 0:
        prob_p1 = prob_p1 / total
        prob_p2 = prob_p2 / total

    c1, c2 = st.columns(2)
    c1.metric(player_1, f"{prob_p1 * 100:.1f}%")
    c2.metric(player_2, f"{prob_p2 * 100:.1f}%")

    pred_df = pd.DataFrame({
        "Player": [player_1, player_2],
        "Win probability": [prob_p1 * 100, prob_p2 * 100]
    })

    fig_pred = px.bar(
        pred_df,
        x="Player",
        y="Win probability",
        text="Win probability",
        title="Predicted win probability"
    )
    fig_pred.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_pred.update_layout(template="plotly_white", yaxis_range=[0, 100])

    st.plotly_chart(fig_pred, use_container_width=True)

st.markdown("---")
st.subheader("Feature comparison")

compare_df = pd.DataFrame({
    "Feature": ["Ranking", "Last 5 form", "Surface win rate", "Overall win rate", "Win vs better players"],
    player_1: [
        rank_map.get(player_1),
        latest_form_map.get(player_1),
        surface_wr_map.get((player_1, surface)),
        overall_wr_map.get(player_1),
        vs_better_map.get(player_1),
    ],
    player_2: [
        rank_map.get(player_2),
        latest_form_map.get(player_2),
        surface_wr_map.get((player_2, surface)),
        overall_wr_map.get(player_2),
        vs_better_map.get(player_2),
    ],
})

st.dataframe(compare_df, use_container_width=True)