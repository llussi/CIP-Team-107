import streamlit as st

pages = [
    st.Page("pages/home.py", title="Overview"),
    st.Page("pages/dashboard.py", title="Dashboard"),
    st.Page("pages/probability_winning.py", title="Performance"),
    st.Page("pages/player_performance.py", title="H2H Analysis"),
    st.Page("pages/prediction.py", title="Tournament Prediction"),
]

pg = st.navigation(pages)

pg.run()
