import re
import os
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


@st.cache_data(show_spinner=False)
def load_csv_folder(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    files = sorted([f for f in os.listdir(path) if f.lower().endswith(".csv")])
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        full_path = os.path.join(path, f)
        try:
            df = pd.read_csv(full_path)
            df["source_file"] = f
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


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


def annual_stats(data: pd.DataFrame, discipline: str, sexe: str, categorie_age: str, score_col: str = "Total") -> pd.DataFrame:
    sub = data[
        (data["discipline"] == discipline)
        & (data["sexe"] == sexe)
        & (data["categorie_age"] == categorie_age)
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
    stats = annual_stats(df, discipline, sexe, categorie_age, score_col=score_col)
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


def plot_athlete_score(data: pd.DataFrame, athlete_name: str, discipline: str, sexe: Optional[str] = None, categorie_age: Optional[str] = None):
    athlete_norm = normalize_athlete_name(athlete_name)
    sub = data[(data["athlete"] == athlete_norm) & (data["discipline"] == discipline)].copy()

    if sexe is not None:
        sub = sub[sub["sexe"] == sexe]
    if categorie_age is not None:
        sub = sub[sub["categorie_age"] == categorie_age]

    if sub.empty:
        return None

    sub = sub.sort_values(["year", "categorie_age"])
    y_col = "Total_normalise_6" if discipline == "10m" else "Total"

    fig = px.line(
        sub,
        x="year",
        y=y_col,
        color="categorie_age",
        markers=True,
        hover_data=["Rank", "discipline", "sexe", "source_pdf"],
        title=f"Évolution du score — {athlete_name} ({discipline})",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Année", yaxis_title="Score")
    return fig


def plot_athlete_rank(data: pd.DataFrame, athlete_name: str, discipline: str, sexe: Optional[str] = None, categorie_age: Optional[str] = None):
    athlete_norm = normalize_athlete_name(athlete_name)
    sub = data[(data["athlete"] == athlete_norm) & (data["discipline"] == discipline)].copy()

    if sexe is not None:
        sub = sub[sub["sexe"] == sexe]
    if categorie_age is not None:
        sub = sub[sub["categorie_age"] == categorie_age]

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


def athlete_summary(data: pd.DataFrame, athlete_name: str, sexe: Optional[str] = None, categorie_age: Optional[str] = None) -> pd.DataFrame:
    athlete_norm = normalize_athlete_name(athlete_name)
    sub = data[data["athlete"] == athlete_norm].copy()

    if sexe is not None:
        sub = sub[sub["sexe"] == sexe]
    if categorie_age is not None:
        sub = sub[sub["categorie_age"] == categorie_age]

    if sub.empty:
        return pd.DataFrame()

    return sub[["about", "year", "discipline", "sexe", "categorie_age", "Rank", "Total", "source_pdf"]].sort_values(["discipline", "year"])


# =========================
# Chargement auto uniquement
# =========================
raw = load_csv_folder("csv")
df = prepare_data(raw) if not raw.empty else pd.DataFrame()

st.title("Analyse CDF — version Streamlit")
st.caption("Page 1 : analyse automatique des CSV du dossier csv/")

if raw.empty:
    st.error("Aucun fichier CSV trouvé dans le dossier 'csv/'.")
    st.stop()

if df.empty:
    st.warning("Les fichiers ont été chargés, mais aucune ligne exploitable n'a été trouvée après nettoyage.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Lignes brutes", len(raw))
col2.metric("Lignes analysables", len(df))
col3.metric("Athlètes", df["athlete"].nunique())
col4.metric("Années", int(df["year"].nunique()) if df["year"].notna().any() else 0)

st.divider()

# =========================
# Filtres
# =========================
st.subheader("Filtres")

f1, f2, f3 = st.columns(3)
selected_sexe = f1.selectbox("Sexe", sorted(df["sexe"].dropna().unique()))
selected_cat = f2.selectbox("Catégorie", sorted(df["categorie_age"].dropna().unique()))
selected_disc = f3.selectbox("Discipline", sorted(df["discipline"].dropna().unique()))

filtered_df = df[
    (df["sexe"] == selected_sexe)
    & (df["categorie_age"] == selected_cat)
    & (df["discipline"] == selected_disc)
].copy()

if filtered_df.empty:
    st.warning("Aucune donnée pour cette combinaison.")
    st.stop()

score_col = "Total_normalise_6" if selected_disc == "10m" else "Total"

st.subheader("Graphique global")
fig_global = plot_annual_scores(df, selected_disc, selected_sexe, selected_cat, score_col=score_col)
if fig_global is not None:
    st.plotly_chart(fig_global, use_container_width=True)
else:
    st.warning("Pas de données pour ce graphique.")

st.divider()

st.subheader("Analyse athlète")
athletes = sorted(filtered_df["athlete"].dropna().unique())
selected_athlete = st.selectbox("Athlète", athletes)

summary = athlete_summary(
    df,
    selected_athlete,
    sexe=selected_sexe,
    categorie_age=selected_cat
)

if not summary.empty:
    st.dataframe(summary, use_container_width=True)

score_fig = plot_athlete_score(
    df,
    selected_athlete,
    selected_disc,
    sexe=selected_sexe,
    categorie_age=selected_cat
)
rank_fig = plot_athlete_rank(
    df,
    selected_athlete,
    selected_disc,
    sexe=selected_sexe,
    categorie_age=selected_cat
)

c1, c2 = st.columns(2)
with c1:
    if score_fig is not None:
        st.plotly_chart(score_fig, use_container_width=True)
    else:
        st.info("Pas de graphique score pour cet athlète.")
with c2:
    if rank_fig is not None:
        st.plotly_chart(rank_fig, use_container_width=True)
    else:
        st.info("Pas de graphique rank pour cet athlète.")

st.divider()

with st.expander("Voir les données filtrées"):
    st.dataframe(filtered_df, use_container_width=True)

st.download_button(
    "Télécharger les données filtrées en CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="cdf_filtre.csv",
    mime="text/csv",
)
