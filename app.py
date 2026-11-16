"""Entry point: `streamlit run app.py`.

Each strategy is one page under ``views/``. Add a page here to add a strategy.
"""

import streamlit as st

st.set_page_config(page_title="Trading strategies", layout="wide")

pages = [
    st.Page("views/quote_vs_cross.py", title="Quote or cross", icon=":material/swap_vert:", default=True),
]

st.navigation(pages).run()
