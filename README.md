# CIP Team 107 — ATP Tennis Match Prediction & Analysis

An interactive Streamlit dashboard for predicting ATP tennis match outcomes and simulating tournament brackets using machine learning, based on historical match data from 2000 to 2026.

## Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Machine Learning Model](#machine-learning-model)
- [Pages](#pages)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

This project applies machine learning to professional tennis data to uncover patterns in player rankings, surface preferences, and match history. It delivers an end-to-end workflow — from data collection and preprocessing to model training and deployment as an interactive web application.

## Research Questions

1. How does win probability vary based on both players' rankings and the ranking difference in ATP matches (2000–2026)?
2. How does a player's performance change depending on the surface and tournament context (Grand Slam vs. other tournaments)?
3. To what extent can historical trends simulate and predict player progression (quarterfinals to final) at the 2026 Australian Open?

## Features

- *Data Dashboard* — Summary statistics, match distribution by surface, top players by wins, yearly trends with COVID-19 impact marker
- *Win Probability Analysis* — Ranking difference vs. win probability charts, heatmap by player/opponent rank group (filterable by surface and series)
- *Head-to-Head Comparison* — Side-by-side player cards, surface/season trends, tournament context breakdowns, H2H records
- *Match Prediction* — Predict win probability between any two players using the trained Random Forest model
- *Tournament Bracket Simulation* — Simulate tournament outcomes from quarterfinals to the final

## Project Structure


├── app.py                          # Main Streamlit entry point (multi-page navigation)
├── DOCUMENTATION.md                # Full project documentation
├── README.md
├── data/
│   └── processed/                  # Processed Parquet data files
├── models/                         # Trained model (atp_match_model.pkl)
├── notebooks/
│   └── Daily update.ipynb          # Data update notebook
├── pages/
│   ├── home.py                     # Overview / introduction page
│   ├── dashboard.py                # Data exploration dashboard
│   ├── probability_winning.py      # Win probability analysis
│   ├── player_performance.py       # Head-to-head player comparison
│   ├── prediction.py               # Match win probability predictor
│   └── prediction_.py              # Tournament bracket simulator
└── src/
    ├── load_data.py                # Data download, cleaning & feature engineering
    └── load_model.py               # Model training pipeline


## Installation

### Prerequisites

- Python 3.9+
- A [Kaggle API key](https://www.kaggle.com/docs/api) configured for kagglehub

### Setup

1. *Clone the repository*
   bash
   git clone https://github.com/your-org/CIP-Team-107.git
   cd CIP-Team-107
   

2. *Install dependencies*
   bash
   pip install streamlit pandas plotly scikit-learn joblib kagglehub
   

3. *Download and process data*
   bash
   python src/load_data.py
   

4. *Train the model*
   bash
   python src/load_model.py
   

## Usage

Launch the Streamlit app:

bash
streamlit run app.py


The app opens in your browser with five pages accessible via the sidebar navigation.

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| ATP Tennis (2000–2026) | [Kaggle — dissfya/atp-tennis-2000-2023daily-pull](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) | 20+ years of ATP match data with player names, rankings, surface, tournament series, round, date, and winner |
| Australian Open 2026 | Tournament-specific dataset | Used for validation against real results |

### Key Variables

- *Numerical:* Rank_1, Rank_2, Rank_diff, win rates, surface win rates, max round reached
- *Categorical:* Surface (Hard, Clay, Grass), Series (Grand Slam, Masters 1000, ATP500, ATP250), Round
- *Target:* Match winner (binary — whether the higher-ranked player wins)

## Machine Learning Model

- *Algorithm:* Random Forest Classifier (n_estimators=100)
- *Features:*
  - Overall win rate (per player)
  - Surface-specific win rate
  - Win rate vs. better/worse-ranked opponents
  - Ranking difference
  - Deepest round reached in similar tournaments
- *Output:* Predicted probability that the higher-ranked player wins
- *Persistence:* Saved via joblib to models/atp_match_model.pkl

## Pages

| Page | Description |
|------|-------------|
| *Overview* | Project introduction, data sources, and research questions |
| *Dashboard* | Total matches, unique surfaces/winners, time range, surface distribution chart, top 10 players, yearly win trends |
| *Performance* | Win probability by ranking difference (line chart), win probability heatmap by rank group (filterable by surface and series) |
| *H2H Analysis* | Player comparison — stats cards, surface/season line charts, tournament bar charts, head-to-head record |
| *Tournament Prediction* | Match-level predictions and full bracket simulation from quarterfinals to final |

## Limitations

- Rankings may have gaps for lower-ranked players; only ATP (men's) data is covered
- Injury, fatigue, and motivation factors are not captured
- Baseline model trained on full dataset without held-out test set
- No hyperparameter tuning performed
- Limited generalization for players with few historical matches

## Future Work

- Hyperparameter tuning and alternative models (XGBoost, Logistic Regression)
- Proper train/test split or time-based cross-validation
- Additional features: serve statistics, break points, match duration, player age
- WTA data integration
- Cloud deployment (Streamlit Cloud, Heroku)
- Model explainability with SHAP values
- Full tournament draw simulation (Round of 128 to Final)

## Contributors

| Member | Contributions |
|--------|---------------|
| *Leon* | Overview page, Dashboard page |
| *Lukas* | Performance page, H2H Analysis page, Tournament Prediction page, data pipeline, model training, and everything else |