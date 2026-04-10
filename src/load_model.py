from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

round_mapping = {
    "1st Round": 1,
    "2nd Round": 2,
    "3rd Round": 3,
    "4th Round": 4,
    "Quarterfinals": 5,
    "Semifinals": 6,
    "The Final": 7
}

def prepare_data_for_model():
    df = pd.read_parquet("data/processed/atp_tennis.parquet")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Is_Grand_Slam"] = df["Series"].eq("Grand Slam")

    p1 = df[["Date", "Year", "Surface", "Series", "Is_Grand_Slam", "Player_1", "Player_2", "Winner", "Round", "Rank_1", "Rank_2"]].copy()
    p1["Player"] = p1["Player_1"]
    p1["Opponent"] = p1["Player_2"]
    p1["Player_rank"] = p1["Rank_1"]
    p1["Opponent_rank"] = p1["Rank_2"]
    p1["Win"] = (p1["Winner"] == p1["Player_1"]).astype(int)

    p2 = df[["Date", "Year", "Surface", "Series", "Is_Grand_Slam", "Player_1", "Player_2", "Winner", "Round", "Rank_1", "Rank_2"]].copy()
    p2["Player"] = p2["Player_2"]
    p2["Opponent"] = p2["Player_1"]
    p2["Player_rank"] = p2["Rank_2"]
    p2["Opponent_rank"] = p2["Rank_1"]
    p2["Win"] = (p2["Winner"] == p2["Player_2"]).astype(int)

    long_df = pd.concat(
        [
            p1[["Date", "Year", "Surface", "Series", "Is_Grand_Slam", "Player", "Opponent", "Player_rank", "Opponent_rank", "Win", "Round"]],
            p2[["Date", "Year", "Surface", "Series", "Is_Grand_Slam", "Player", "Opponent", "Player_rank", "Opponent_rank", "Win", "Round"]],
        ],
        ignore_index=True,
    )

    long_df["Round_num"] = long_df["Round"].map(round_mapping)

    df_model = df[["Year", "Surface", "Series", "Player_1", "Player_2", "Winner", "Round", "Rank_1", "Rank_2"]].copy()

    mask = df_model["Rank_2"] < df_model["Rank_1"]

    # swap jugadores
    df_model.loc[mask, ["Player_1", "Player_2"]] = df_model.loc[mask, ["Player_2", "Player_1"]].values

    # swap rankings
    df_model.loc[mask, ["Rank_1", "Rank_2"]] = df_model.loc[mask, ["Rank_2", "Rank_1"]].values
    df_model["Win"] = (df_model["Winner"] == df_model["Player_1"]).astype(int)

    return long_df.dropna(subset=["Player", "Opponent", "Surface", "Year", "Round"]), df_model

def prepare_player_stats(df: pd.DataFrame):
    player_stats = (
        df.groupby("Player")
        .agg(
            Matches=("Win", "count"),
            Wins=("Win", "sum")
        )
        .reset_index()
    )

    player_stats["Win_rate"] = player_stats["Wins"] / player_stats["Matches"]
    return player_stats

def prepare_player_surface_stats(df: pd.DataFrame):
    surface_stats = (
        df.groupby(["Player", "Surface"])
        .agg(
            Matches=("Win", "count"),
            Wins=("Win", "sum")
        )
        .reset_index()
    )

    surface_stats["Surface_win_rate"] = surface_stats["Wins"] / surface_stats["Matches"]
    return surface_stats

def prepare_vs_better_stats(df: pd.DataFrame):
    df["Better_opponent"] = df["Opponent_rank"] < df["Player_rank"]

    vs_better = (
        df[df["Better_opponent"]]
        .groupby("Player")
        .agg(
            Matches=("Win", "count"),
            Wins=("Win", "sum")
        )
        .reset_index()
    )

    vs_better["Win_vs_better"] = vs_better["Wins"] / vs_better["Matches"]
    return vs_better

def prepare_vs_worse_stats(df: pd.DataFrame):
    df["Worse_opponent"] = df["Opponent_rank"] > df["Player_rank"]

    vs_worse = (
        df[df["Worse_opponent"]]
        .groupby("Player")
        .agg(
            Matches=("Win", "count"),
            Wins=("Win", "sum")
        )
        .reset_index()
    )

    vs_worse["Win_vs_worse"] = vs_worse["Wins"] / vs_worse["Matches"]
    return vs_worse

def prepare_last5_form(data):

    data = data.sort_values(by="Date")

    data["Last5_win_rate"] = (
        data
        .groupby("Player")["Win"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )

    return data

def prepare_tournament_results(df: pd.DataFrame):
    tournament_results = (
        df.groupby(["Player", "Series", "Surface"])
        .agg(Max_round=("Round_num", "max"))
        .reset_index()
    )
    return tournament_results

def create_model(
        df: pd.DataFrame,
        df_player_stats_: pd.DataFrame,
        df_surface_stats_: pd.DataFrame,
        df_vs_better_stats_: pd.DataFrame,
        df_vs_worse_stats_: pd.DataFrame,
        df_last5_form_: pd.DataFrame,
        df_tournament_results_: pd.DataFrame
):
    df = df.merge(
        df_player_stats_[["Player", "Win_rate"]].rename(columns={"Win_rate": "Win_rate_player_1"}),
        left_on="Player_1",
        right_on="Player",
        how="left"
    ).drop(columns=["Player"])

    df = df.merge(
        df_player_stats_[["Player", "Win_rate"]].rename(columns={"Win_rate": "Win_rate_player_2"}),
        left_on="Player_2",
        right_on="Player",
        how="left"
    ).drop(columns=["Player"])

    df = df.merge(
        df_surface_stats_[["Player", "Surface", "Surface_win_rate"]].rename(
            columns={"Surface_win_rate": "Surface_win_rate_player_1"}),
        left_on=["Player_1", "Surface"],
        right_on=["Player", "Surface"],
        how="left"
    ).drop(columns=["Player"])

    df = df.merge(
        df_surface_stats_[["Player", "Surface", "Surface_win_rate"]].rename(
            columns={"Surface_win_rate": "Surface_win_rate_player_2"}),
        left_on=["Player_2", "Surface"],
        right_on=["Player", "Surface"],
        how="left"
    ).drop(columns=["Player"])

    df = df.merge(
        df_vs_worse_stats_[["Player", "Win_vs_worse"]].rename(columns={"Win_vs_worse": "Lose_vs_worse_player_1"}),
        left_on="Player_1",
        right_on="Player",
        how="left").drop(columns=["Player"])

    df = df.merge(
        df_vs_better_stats_[["Player", "Win_vs_better"]].rename(columns={"Win_vs_better": "Win_vs_better_player_2"}),
        left_on="Player_2",
        right_on="Player",
        how="left").drop(columns=["Player"])

    # df = df.merge(
    #     df_last5_form_[["Player", "Last5_win_rate"]].rename(columns={"Last5_win_rate": "Last5_win_rate_player_1"}),
    #     left_on="Player_1",
    #     right_on="Player",
    #     how="left").drop(columns=["Player"])
    #
    # df = df.merge(
    #     df_last5_form_[["Player", "Last5_win_rate"]].rename(columns={"Last5_win_rate": "Last5_win_rate_player_2"}),
    #     left_on="Player_2",
    #     right_on="Player",
    #     how="left").drop(columns=["Player"])
    df = df.merge(
        df_tournament_results_[["Player", "Series", "Surface", "Max_round"]].rename(columns={"Max_round": "Max_round_player_1"}),
        left_on=["Player_1","Series","Surface"],
        right_on=["Player","Series","Surface"],
        how="left").drop(columns=["Player"])

    df = df.merge(
        df_tournament_results_[["Player", "Series", "Surface", "Max_round"]].rename(columns={"Max_round": "Max_round_player_2"}),
        left_on=["Player_2","Series","Surface"],
        right_on=["Player","Series","Surface"],
        how="left").drop(columns=["Player"])

    df["Rank_diff"] = df["Rank_1"] - df["Rank_2"]

    X = df[
        [
            "Win_rate_player_1",
            "Win_rate_player_2",
            "Surface_win_rate_player_1",
            "Surface_win_rate_player_2",
            "Lose_vs_worse_player_1",
            "Win_vs_better_player_2",
            "Rank_diff",
            "Max_round_player_1",
            "Max_round_player_2"

        ]
    ]

    y = df["Win"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    dump(model, "models/atp_match_model_.pkl")

    return df


if __name__ == "__main__":
    print("Downloading atp tennis data")
    df_prepared, df_model = prepare_data_for_model()

    df_player_stats = prepare_player_stats(df_prepared)

    df_surface_stats = prepare_player_surface_stats(df_prepared)

    df_vs_better_stats = prepare_vs_better_stats(df_prepared)

    df_vs_worse_stats = prepare_vs_worse_stats(df_prepared)

    df_last5_form = prepare_last5_form(df_prepared)

    df_tournament_results = prepare_tournament_results(df_last5_form)

    create_model(df_model,df_player_stats,df_surface_stats,df_vs_better_stats,df_vs_worse_stats,df_last5_form,df_tournament_results)



    print("Finish downloading atp tennis data")