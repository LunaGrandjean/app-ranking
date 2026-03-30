import streamlit as st

st.set_page_config(page_title="App FFTir", layout="wide")

st.title("App FFTir")
st.caption("Accueil de l'application Streamlit")

st.markdown(
    """
    Cette app contient 2 pages :

    - **Analyse CDF** : exploration des CSV, statistiques annuelles et analyse par athlète.
    - **Ranking App** : calcul du tableau final à partir du fichier Excel FFTir.

    Utilise le menu de gauche pour naviguer entre les pages.
    """
)

c1, c2 = st.columns(2)
with c1:
    st.info("Ouvre **pages/1_Analyse_CDF.py** pour l'analyse des données CDF.")
with c2:
    st.info("Ouvre **pages/2_Ranking_App.py** pour le calcul du ranking final.")
