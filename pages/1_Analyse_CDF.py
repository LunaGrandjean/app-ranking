import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Analyse CDF", layout="wide")

EXPECTED_COLS = [
    "Rank", "Phase", "about", "date", "competition", "Location", "Description", "Total",
    "score_finale", "mouches", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12",
    "P1", "P2", "P3", "P4", "P5", "Épreuve", "source_pdf"
]
SERIES_COLS = [f"S{i}" for i in range(1, 13)]
NUM_COLS = ["Rank", "Total", "score_finale"] + SERIES_COLS + [f"P{i}" for i in range(1, 6)]

def load_csv_folder(path):
    files = [f for f in os.listdir(path) if f.endswith(".csv")]

    if not files:
        return None

    dfs = []
    for f in files:
        full_path = os.path.join(path, f)
        df = pd.read_csv(full_path)
        df["source_file"] = f
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

# ---- UI ----
path = "csv"

if os.path.exists(path):
    df = load_csv_folder(path)

    if df is not None:
        st.success(f"{len(df)} lignes chargées depuis {len(df['source_file'].unique())} fichiers")
        st.dataframe(df.head())
    else:
        st.warning("Aucun CSV trouvé dans le dossier.")
else:
    st.error("Le dossier csv/ n'existe pas.")


def normalize_athlete_name(name: str) -> str:
    name = str(name).strip().upper()
    name = re.sub(r"\s+", " ", name)
    return name


def parse_event_info(pdf_name: str) -> pd.Series:
    name = str(pdf_name).lower()

    discipline = "50m" if "50" in name else "10m"

    if "women" in name:
        sexe = "Women"
    elif "men" in name:
        sexe = "Men"
    else:
        sexe = "unknown"

    if "cadet" in name:
        categorie_age = "Cadet"
    elif "junior" in name:
        categorie_age = "Junior"
    else:
        categorie_age = "Senior"

    return pd.Series([discipline, sexe, categorie_age])


@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()

    df = raw.copy()

    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    for c in ["about", "Phase", "source_pdf", "Description", "Épreuve"]:
        df[c] = df[c].astype(str).str.strip()

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    df[["discipline", "sexe", "categorie_age"]] = df["source_pdf"].apply(parse_event_info)
    df["nb_series"] = df[SERIES_COLS].notna().sum(axis=1)
    df["Total_normalise_6"] = df["Total"]

    mask_10m = (df["discipline"] == "10m") & (df["nb_series"] > 0)
    df.loc[mask_10m, "Total_normalise_6"] = (
        df.loc[mask_10m, "Total"] * 6 / df.loc[mask_10m, "nb_series"]
    )

    df["athlete"] = df["about"].apply(normalize_athlete_name)

    df = df[
        (df["Rank"].notna())
        & (df["Rank"] > 0)
        & (df["Total"].notna())
        & (df["Total"] > 0)
        & (df["nb_series"] > 0)
    ].copy()

    return df


def annual_stats(data: pd.DataFrame, discipline: str, sexe: str, score_col: str = "Total") -> pd.DataFrame:
    sub = data[
        (data["discipline"] == discipline)
        & (data["sexe"] == sexe)
    ].copy()

    sub = sub.dropna(subset=["Rank", score_col, "year"])
    sub["Rank"] = pd.to_numeric(sub["Rank"], errors="coerce")
    sub = sub.dropna(subset=["Rank"])
    sub["Rank"] = sub["Rank"].astype(int)

    out = []
    for year, g in sub.groupby("year"):
        g = g.sort_values("Rank")
        out.append(
            {
                "year": year,
                "first_score": g.loc[g["Rank"] == 1, score_col].mean(),
                "podium_mean": g[g["Rank"] <= 3][score_col].mean(),
                "top10_mean": g[g["Rank"] <= 10][score_col].mean(),
                "rank10_score": g.loc[g["Rank"] == 10, score_col].mean(),
            }
        )

    return pd.DataFrame(out).sort_values("year") if out else pd.DataFrame()


def plot_annual_scores(df: pd.DataFrame, discipline: str, sexe: str, categorie_age: str, score_col: str = "Total"):
    dff = df[
        (df["discipline"] == discipline)
        & (df["sexe"] == sexe)
        & (df["categorie_age"] == categorie_age)
    ].copy()

    stats = annual_stats(dff, discipline, sexe, score_col=score_col)
    if stats.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=stats["year"],
            y=stats["rank10_score"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stats["year"],
            y=stats["first_score"],
            mode="lines",
            fill="tonexty",
            name="Couloir 1er-10e",
            hovertemplate="Année=%{x}<br>1er=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stats["year"],
            y=stats["first_score"],
            mode="lines+markers",
            name="1er",
            hovertemplate="Année=%{x}<br>1er=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stats["year"],
            y=stats["podium_mean"],
            mode="lines+markers",
            name="Moyenne podium",
            hovertemplate="Année=%{x}<br>Podium=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stats["year"],
            y=stats["top10_mean"],
            mode="lines+markers",
            name="Moyenne top 10",
            hovertemplate="Année=%{x}<br>Top10=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Évolution annuelle des scores — {discipline} / {sexe} / {categorie_age}",
        xaxis_title="Année",
        yaxis_title="Score",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_athlete_score(data: pd.DataFrame, athlete_name: str, discipline: str, sexe: Optional[str] = None):
    athlete_norm = normalize_athlete_name(athlete_name)
    sub = data[(data["athlete"] == athlete_norm) & (data["discipline"] == discipline)].copy()
    if sexe is not None:
        sub = sub[sub["sexe"] == sexe]
    if sub.empty:
        return None

    sub = sub.sort_values(["year", "categorie_age"])
    fig = px.line(
        sub,
        x="year",
        y="Total",
        color="categorie_age",
        markers=True,
        hover_data=["Rank", "discipline", "sexe", "source_pdf"],
        title=f"Évolution du score — {athlete_name} ({discipline})",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Année", yaxis_title="Score")
    return fig


def plot_athlete_rank(data: pd.DataFrame, athlete_name: str, discipline: str, sexe: Optional[str] = None):
    athlete_norm = normalize_athlete_name(athlete_name)
    sub = data[(data["athlete"] == athlete_norm) & (data["discipline"] == discipline)].copy()
    if sexe is not None:
        sub = sub[sub["sexe"] == sexe]
    if sub.empty:
        return None

    sub = sub.sort_values(["year", "categorie_age"])
    fig = px.line(
        sub,
        x="year",
        y="Rank",
        color="categorie_age",
        markers=True,
        hover_data=["Total", "discipline", "sexe", "source_pdf"],
        title=f"Évolution du rank — {athlete_name} ({discipline})",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Année", yaxis_title="Rank")
    fig.update_yaxes(autorange="reversed")
    return fig


def athlete_summary(data: pd.DataFrame, athlete_name: str) -> pd.DataFrame:
    athlete_norm = normalize_athlete_name(athlete_name)
    sub = data[data["athlete"] == athlete_norm].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub[["about", "year", "discipline", "sexe", "categorie_age", "Rank", "Total", "source_pdf"]].sort_values(["discipline", "year"])


def top_athletes(data: pd.DataFrame, categorie_age: str, discipline: str, n: int = 15) -> pd.DataFrame:
    sub = (
        data[(data["categorie_age"] == categorie_age) & (data["discipline"] == discipline)]
        .dropna(subset=["Rank"])
        .sort_values(["Rank", "Total"], ascending=[True, False])
        .drop_duplicates("athlete")
        .head(n)
    )
    return sub[["athlete", "year", "Rank", "Total", "sexe", "source_pdf"]]


st.title("Analyse CDF — version Streamlit")
st.caption("Page 1 : analyse des CSV extraits du notebook CDF.ipynb")

with st.sidebar:
    st.header("Chargement des données")
    source_mode = st.radio(
        "Source des CSV",
        ["Dossier local csv", "Upload manuel"],
        index=0,
    )

    if source_mode == "Dossier local csv":
        folder = st.text_input("Chemin du dossier CSV", value="csv")
        raw = load_data_from_folder(folder)
    else:
        uploaded_files = st.file_uploader(
            "Importer un ou plusieurs CSV",
            type=["csv"],
            accept_multiple_files=True,
        )
        raw = load_data_from_uploads(uploaded_files) if uploaded_files else pd.DataFrame()


df = prepare_data(raw) if not raw.empty else pd.DataFrame()

if raw.empty:
    st.info("Aucun fichier CSV chargé. Place tes fichiers dans un dossier `csv/` à côté du script, ou importe-les manuellement dans la barre latérale.")
    st.stop()

if df.empty:
    st.warning("Les fichiers ont été chargés, mais aucune ligne exploitable n'a été trouvée après nettoyage.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Lignes brutes", len(raw))
col2.metric("Lignes analysables", len(df))
col3.metric("Athlètes", df["athlete"].nunique())
col4.metric("Années", int(df["year"].nunique()) if df["year"].notna().any() else 0)

with st.expander("Aperçu des données préparées"):
    st.dataframe(df.head(50), use_container_width=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "Scores annuels",
    "Analyse athlète",
    "Top athlètes",
    "Données",
])

with tab1:
    st.subheader("Évolution annuelle des scores")
    c1, c2, c3 = st.columns(3)
    discipline = c1.selectbox("Discipline", sorted(df["discipline"].dropna().unique()), key="annual_discipline")
    sexe = c2.selectbox("Sexe", sorted(df["sexe"].dropna().unique()), key="annual_sexe")
    categorie = c3.selectbox("Catégorie", sorted(df["categorie_age"].dropna().unique()), key="annual_cat")

    score_col = "Total_normalise_6" if discipline == "10m" else "Total"
    fig = plot_annual_scores(df, discipline, sexe, categorie, score_col=score_col)
    if fig is None:
        st.warning("Pas de données pour cette combinaison.")
    else:
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Analyse par athlète")
    athletes = sorted(df["athlete"].dropna().unique())
    athlete = st.selectbox("Athlète", athletes)
    sex_filter_options = ["Tous"] + sorted(df[df["athlete"] == athlete]["sexe"].dropna().unique().tolist())
    sex_filter = st.selectbox("Filtre sexe", sex_filter_options)
    sex_filter_value = None if sex_filter == "Tous" else sex_filter

    summary = athlete_summary(df, athlete)
    if sex_filter_value is not None and not summary.empty:
        summary = summary[summary["sexe"] == sex_filter_value]

    if summary.empty:
        st.warning("Athlète non trouvé.")
    else:
        st.dataframe(summary, use_container_width=True)
        available_disciplines = sorted(summary["discipline"].dropna().unique())
        for disc in available_disciplines:
            score_fig = plot_athlete_score(df, athlete, disc, sexe=sex_filter_value)
            rank_fig = plot_athlete_rank(df, athlete, disc, sexe=sex_filter_value)
            if score_fig is not None:
                st.plotly_chart(score_fig, use_container_width=True)
            if rank_fig is not None:
                st.plotly_chart(rank_fig, use_container_width=True)

with tab3:
    st.subheader("Top athlètes")
    c1, c2, c3 = st.columns(3)
    top_cat = c1.selectbox("Catégorie âge", sorted(df["categorie_age"].dropna().unique()), key="top_cat")
    top_disc = c2.selectbox("Discipline", sorted(df["discipline"].dropna().unique()), key="top_disc")
    top_n = c3.slider("Nombre d'athlètes", min_value=5, max_value=30, value=15)

    st.dataframe(top_athletes(df, top_cat, top_disc, top_n), use_container_width=True)

with tab4:
    st.subheader("Exploration libre")
    years = sorted([int(y) for y in df["year"].dropna().unique()])
    selected_years = st.multiselect("Années", years, default=years)
    selected_sexes = st.multiselect("Sexes", sorted(df["sexe"].dropna().unique()), default=sorted(df["sexe"].dropna().unique()))
    selected_cats = st.multiselect("Catégories", sorted(df["categorie_age"].dropna().unique()), default=sorted(df["categorie_age"].dropna().unique()))
    selected_disc = st.multiselect("Disciplines", sorted(df["discipline"].dropna().unique()), default=sorted(df["discipline"].dropna().unique()))

    filtered = df[
        df["year"].isin(selected_years)
        & df["sexe"].isin(selected_sexes)
        & df["categorie_age"].isin(selected_cats)
        & df["discipline"].isin(selected_disc)
    ].copy()

    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Télécharger les données filtrées en CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="cdf_filtre.csv",
        mime="text/csv",
    )
