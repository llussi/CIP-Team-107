from pathlib import Path
import pandas as pd
import kagglehub

def round_change():


def download_atp_data():
    path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
    df = pd.read_csv(Path(path) / "atp_tennis.csv", low_memory=False)

    df = df[df['Surface'] != 'Carpet']

    df['Player_1'] = df['Player_1'].str.rstrip()
    df['Player_2'] = df['Player_2'].str.rstrip()
    df['Winner'] = df['Winner'].str.rstrip()

    df["Loser"] = df.apply(
        lambda row: row["Player_2"] if row["Winner"] == row["Player_1"] else row["Player_1"],
        axis=1
    )
    df["Better_rank_winner"] = (
            ((df["Rank_1"] < df["Rank_2"]) & (df["Winner"] == df["Player_1"])) |
            ((df["Rank_2"] < df["Rank_1"]) & (df["Winner"] == df["Player_2"]))
    )

    df["Rank_diff"] = (df["Rank_1"] - df["Rank_2"]).abs()

    # Hacer que todos los Torneos se llamen iguales
    df['Series'] = df['Series'].replace({
        'International Gold': 'ATP500',
        'International': 'ATP250',
        'Masters': 'Masters 1000'
    })

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_dir / "atp_tennis_raw.csv"
    df.to_csv(output_file, index=False)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)



    df.to_parquet(output_dir / "atp_tennis.parquet", index=False)

    return df

def prepare_player_surface_stats(df: pd.DataFrame):
    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Year"] = data["Date"].dt.year

    # Fila para Player_1
    p1 = data[["Year", "Surface", "Player_1", "Winner"]].copy()
    p1["Player"] = p1["Player_1"]
    p1["Win"] = (p1["Winner"] == p1["Player_1"]).astype(int)
    p1 = p1.drop(columns=["Player_1", "Winner"])

    # Fila para Player_2
    p2 = data[["Year", "Surface", "Player_2", "Winner"]].copy()
    p2["Player"] = p2["Player_2"]
    p2["Win"] = (p2["Winner"] == p2["Player_2"]).astype(int)
    p2 = p2.drop(columns=["Player_2", "Winner"])

    # Unir ambas
    player_matches = pd.concat([p1, p2], ignore_index=True)

    # Agrupar por jugador, año y superficie
    stats = (
        player_matches
        .groupby(["Player", "Year", "Surface" ], as_index=False)
        .agg(
            Matches=("Win", "count"),
            Wins=("Win", "sum")
        )
    )

    stats["Win_pct"] = stats["Wins"] / stats["Matches"] * 100

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_dir / "stats_players.csv"
    stats.to_csv(output_file, index=False)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats.to_parquet(output_dir / "stats_players.parquet", index=False)


if __name__ == "__main__":
    print("Downloading atp tennis data")
    df = download_atp_data()
    prepare_player_surface_stats(df)
    print("Finish downloading atp tennis data")