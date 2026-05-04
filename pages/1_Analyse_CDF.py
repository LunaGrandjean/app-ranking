import re
import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================
# Configuration
# =========================

st.set_page_config(page_title="Analyse CDF", layout="wide")

CSV_FOLDER = "csv"
WORLD_CSV_PATH = "all_world_cups_clean.csv"

EXPECTED_COLS = [
    "Rank", "Phase", "about", "date", "competition", "Location", "Description", "Total",
    "score_finale", "mouches", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8",
    "S9", "S10", "S11", "S12", "P1", "P2", "P3", "P4", "P5", "Épreuve", "source_pdf"
]

SERIES_COLS = [f"S{i}" for i in range(1, 13)]
NUM_COLS = ["Rank", "Total", "score_finale"] + SERIES_COLS + [f"P{i}" for i in range(1, 6)]

ATHLETES_SUIVIS = [
    "MULLER Océanne",
    "KRYZS Lucas",
    "DUTENDAS Dimitri",
    "BAUDOUIN Brian",
    "GIRARD Agathe",
    "HERBULOT Manon",
    "CANESTRELLI Julia",
    "GOMEZ Judith",
    "AUFRERE Romain",
    "BORDET Jade",
    "MONIER Jérémy",
    "D'HALLUIN Michael",
]


# =========================
# Chargement données France
# =========================

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


# =========================
# Chargement données monde
# =========================

@st.cache_data(show_spinner=False)
def load_world_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)

    required_cols = ["year", "event", "rank", "qualification"]

    for c in required_cols:
        if c not in df.columns:
            return pd.DataFrame()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["qualification"] = pd.to_numeric(df["qualification"], errors="coerce")
    df["final"] = pd.to_numeric(df.get("final", np.nan), errors="coerce")

    df["event"] = df["event"].astype(str).str.strip()

    df = df.dropna(subset=["year", "rank", "qualification", "event"]).copy()
    df["year"] = df["year"].astype(int)
    df["rank"] = df["rank"].astype(int)

    return df


def world_event_name(discipline: str, sexe: str) -> Optional[str]:
    if discipline == "10m":
        return f"10m Air Rifle {sexe}"

    if discipline == "50m":
        return f"50m Rifle 3 Positions {sexe}"

    return None


def world_annual_stats(
    world_df: pd.DataFrame,
    discipline: str,
    sexe: str,
    score_col: str = "qualification"
) -> pd.DataFrame:
    if world_df.empty:
        return pd.DataFrame()

    event = world_event_name(discipline, sexe)

    if event is None:
        return pd.DataFrame()

    sub = world_df[world_df["event"] == event].copy()

    if sub.empty:
        return pd.DataFrame()

    sub = sub.dropna(subset=["year", "rank", score_col])

    out = []

    for year, g in sub.groupby("year"):
        g = g.sort_values("rank")

        out.append({
            "year": year,
            "world_first_score": g.loc[g["rank"] == 1, score_col].mean(),
            "world_podium_mean": g.loc[g["rank"] <= 3, score_col].mean(),
            "world_top10_mean": g.loc[g["rank"] <= 10, score_col].mean(),
            "world_rank10_score": g.loc[g["rank"] == 10, score_col].mean(),
        })

    return pd.DataFrame(out).sort_values("year") if out else pd.DataFrame()


# =========================
# Stats France
# =========================

def annual_stats(
    data: pd.DataFrame,
    discipline: str,
    sexe: str,
    categorie_age: str,
    score_col: str = "Total"
) -> pd.DataFrame:
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

        out.append({
            "year": year,
            "france_first_score": g.loc[g["Rank"] == 1, score_col].mean(),
            "france_podium_mean": g.loc[g["Rank"] <= 3, score_col].mean(),
            "france_top10_mean": g.loc[g["Rank"] <= 10, score_col].mean(),
            "france_rank10_score": g.loc[g["Rank"] == 10, score_col].mean(),
        })

    return pd.DataFrame(out).sort_values("year") if out else pd.DataFrame()


# =========================
# Graphiques globaux
# =========================

def plot_annual_scores(
    df: pd.DataFrame,
    discipline: str,
    sexe: str,
    categorie_age: str,
    score_col: str = "Total"
):
    stats = annual_stats(df, discipline, sexe, categorie_age, score_col=score_col)

    if stats.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_rank10_score"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_first_score"],
        mode="lines",
        fill="tonexty",
        name="Couloir France 1er-10e",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_first_score"],
        mode="lines+markers",
        name="France - 1er",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_podium_mean"],
        mode="lines+markers",
        name="France - moyenne podium",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_top10_mean"],
        mode="lines+markers",
        name="France - moyenne top 10",
    ))

    fig.update_layout(
        title=f"Évolution annuelle des scores — {discipline} / {sexe} / {categorie_age}",
        xaxis_title="Année",
        yaxis_title="Score",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_france_vs_world(
    france_df: pd.DataFrame,
    world_df: pd.DataFrame,
    discipline: str,
    sexe: str,
    categorie_age: str,
    score_col_france: str = "Total",
):
    france_stats = annual_stats(
        france_df,
        discipline,
        sexe,
        categorie_age,
        score_col=score_col_france
    )

    world_stats = world_annual_stats(
        world_df,
        discipline,
        sexe,
        score_col="qualification"
    )

    if france_stats.empty or world_stats.empty:
        return None, france_stats, world_stats

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=france_stats["year"],
        y=france_stats["france_first_score"],
        mode="lines+markers",
        name="France - 1er CDF",
    ))

    fig.add_trace(go.Scatter(
        x=france_stats["year"],
        y=france_stats["france_podium_mean"],
        mode="lines+markers",
        name="France - moyenne podium CDF",
    ))

    fig.add_trace(go.Scatter(
        x=france_stats["year"],
        y=france_stats["france_top10_mean"],
        mode="lines+markers",
        name="France - moyenne top 10 CDF",
    ))

    fig.add_trace(go.Scatter(
        x=world_stats["year"],
        y=world_stats["world_first_score"],
        mode="lines+markers",
        name="Monde - 1er Coupe du monde",
        line=dict(dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=world_stats["year"],
        y=world_stats["world_podium_mean"],
        mode="lines+markers",
        name="Monde - moyenne podium",
        line=dict(dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=world_stats["year"],
        y=world_stats["world_top10_mean"],
        mode="lines+markers",
        name="Monde - moyenne top 10",
        line=dict(dash="dash"),
    ))

    fig.update_layout(
        title=f"Comparaison France vs Monde — {discipline} / {sexe} / {categorie_age}",
        xaxis_title="Année",
        yaxis_title="Score qualification",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig, france_stats, world_stats


# =========================
# Graphiques athlètes
# =========================

def plot_athlete_vs_category_by_year(
    data: pd.DataFrame,
    world_df: pd.DataFrame,
    athlete_name: str,
    discipline: str,
    score_col: str = "Total",
):
    athlete_norm = normalize_athlete_name(athlete_name)

    athlete_rows = data[
        (data["athlete"] == athlete_norm)
        & (data["discipline"] == discipline)
        & (data["year"].between(2016, 2026))
    ].copy()

    if athlete_rows.empty:
        return None, pd.DataFrame()

    athlete_rows = athlete_rows.sort_values("year")

    rows = []

    for _, athlete_row in athlete_rows.iterrows():
        year = athlete_row["year"]
        sexe = athlete_row["sexe"]
        categorie = athlete_row["categorie_age"]

        same_context = data[
            (data["year"] == year)
            & (data["discipline"] == discipline)
            & (data["sexe"] == sexe)
            & (data["categorie_age"] == categorie)
        ].copy()

        same_context = same_context.dropna(subset=["Rank", score_col])
        same_context["Rank"] = same_context["Rank"].astype(int)

        world_context = world_annual_stats(
            world_df,
            discipline,
            sexe,
            score_col="qualification"
        )

        world_year = world_context[world_context["year"] == year]

        rows.append({
            "year": year,
            "athlete_score": athlete_row[score_col],
            "athlete_rank": athlete_row["Rank"],
            "categorie_age": categorie,
            "sexe": sexe,
            "france_first_score": same_context.loc[same_context["Rank"] == 1, score_col].mean(),
            "france_podium_mean": same_context.loc[same_context["Rank"] <= 3, score_col].mean(),
            "france_top10_mean": same_context.loc[same_context["Rank"] <= 10, score_col].mean(),
            "france_rank10_score": same_context.loc[same_context["Rank"] == 10, score_col].mean(),
            "world_first_score": world_year["world_first_score"].mean() if not world_year.empty else np.nan,
            "world_podium_mean": world_year["world_podium_mean"].mean() if not world_year.empty else np.nan,
            "world_top10_mean": world_year["world_top10_mean"].mean() if not world_year.empty else np.nan,
        })

    stats = pd.DataFrame(rows).sort_values("year")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_rank10_score"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_first_score"],
        mode="lines",
        fill="tonexty",
        name="Couloir France 1er-10e catégorie réelle",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_first_score"],
        mode="lines+markers",
        name="France - 1er catégorie",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_podium_mean"],
        mode="lines+markers",
        name="France - moyenne podium catégorie",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["france_top10_mean"],
        mode="lines+markers",
        name="France - moyenne top 10 catégorie",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["world_first_score"],
        mode="lines+markers",
        name="Monde - 1er",
        line=dict(dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["world_podium_mean"],
        mode="lines+markers",
        name="Monde - moyenne podium",
        line=dict(dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["world_top10_mean"],
        mode="lines+markers",
        name="Monde - moyenne top 10",
        line=dict(dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["athlete_score"],
        mode="lines+markers+text",
        name=athlete_name,
        text=stats["categorie_age"],
        textposition="top center",
        hovertemplate=(
            "Année=%{x}<br>"
            "Score=%{y:.2f}<br>"
            "Catégorie=%{text}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=f"Évolution 2016-2026 — {athlete_name} vs France et Monde",
        xaxis_title="Année",
        yaxis_title="Score",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig, stats


def plot_athlete_score(
    data: pd.DataFrame,
    world_df: pd.DataFrame,
    athlete_name: str,
    discipline: str,
    sexe: Optional[str] = None,
    categorie_age: Optional[str] = None,
):
    athlete_norm = normalize_athlete_name(athlete_name)

    sub = data[
        (data["athlete"] == athlete_norm)
        & (data["discipline"] == discipline)
    ].copy()

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

    if sexe is not None and not world_df.empty:
        world_stats = world_annual_stats(world_df, discipline, sexe)

        if not world_stats.empty:
            fig.add_trace(go.Scatter(
                x=world_stats["year"],
                y=world_stats["world_first_score"],
                mode="lines+markers",
                name="Monde - 1er",
                line=dict(dash="dash"),
            ))

            fig.add_trace(go.Scatter(
                x=world_stats["year"],
                y=world_stats["world_podium_mean"],
                mode="lines+markers",
                name="Monde - moyenne podium",
                line=dict(dash="dash"),
            ))

            fig.add_trace(go.Scatter(
                x=world_stats["year"],
                y=world_stats["world_top10_mean"],
                mode="lines+markers",
                name="Monde - moyenne top 10",
                line=dict(dash="dash"),
            ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Année",
        yaxis_title="Score",
        hovermode="x unified",
    )

    return fig


def plot_athlete_rank(
    data: pd.DataFrame,
    athlete_name: str,
    discipline: str,
    sexe: Optional[str] = None,
    categorie_age: Optional[str] = None,
):
    athlete_norm = normalize_athlete_name(athlete_name)

    sub = data[
        (data["athlete"] == athlete_norm)
        & (data["discipline"] == discipline)
    ].copy()

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

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Année",
        yaxis_title="Rank"
    )

    fig.update_yaxes(autorange="reversed")

    return fig


def athlete_summary(
    data: pd.DataFrame,
    athlete_name: str,
    sexe: Optional[str] = None,
    categorie_age: Optional[str] = None,
) -> pd.DataFrame:
    athlete_norm = normalize_athlete_name(athlete_name)

    sub = data[data["athlete"] == athlete_norm].copy()

    if sexe is not None:
        sub = sub[sub["sexe"] == sexe]

    if categorie_age is not None:
        sub = sub[sub["categorie_age"] == categorie_age]

    if sub.empty:
        return pd.DataFrame()

    return sub[
        ["about", "year", "discipline", "sexe", "categorie_age", "Rank", "Total", "Total_normalise_6", "source_pdf"]
    ].sort_values(["discipline", "year"])


# =========================
# Interface Streamlit
# =========================

raw = load_csv_folder(CSV_FOLDER)
df = prepare_data(raw) if not raw.empty else pd.DataFrame()
world_df = load_world_csv(WORLD_CSV_PATH)

st.title("Analyse CDF — France vs Monde")
st.caption("Analyse automatique des CSV du dossier csv/ avec comparaison aux niveaux mondiaux.")

if raw.empty:
    st.error("Aucun fichier CSV trouvé dans le dossier 'csv/'.")
    st.stop()

if df.empty:
    st.warning("Les fichiers ont été chargés, mais aucune ligne exploitable n'a été trouvée après nettoyage.")
    st.stop()

if world_df.empty:
    st.warning("Le fichier mondial n'a pas été trouvé ou n'est pas exploitable. Les graphiques France fonctionneront, mais pas la comparaison Monde.")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Lignes brutes France", len(raw))
col2.metric("Lignes analysables France", len(df))
col3.metric("Athlètes France", df["athlete"].nunique())
col4.metric("Années France", int(df["year"].nunique()) if df["year"].notna().any() else 0)
col5.metric("Lignes Monde", len(world_df))

st.divider()


# =========================
# Filtres
# =========================

st.subheader("Filtres")

f1, f2, f3 = st.columns(3)

selected_sexe = f1.selectbox(
    "Sexe",
    sorted(df["sexe"].dropna().unique())
)

selected_cat = f2.selectbox(
    "Catégorie France",
    sorted(df["categorie_age"].dropna().unique())
)

selected_disc = f3.selectbox(
    "Discipline",
    sorted(df["discipline"].dropna().unique())
)

filtered_df = df[
    (df["sexe"] == selected_sexe)
    & (df["categorie_age"] == selected_cat)
    & (df["discipline"] == selected_disc)
].copy()

if filtered_df.empty:
    st.warning("Aucune donnée pour cette combinaison.")
    st.stop()

score_col = "Total_normalise_6" if selected_disc == "10m" else "Total"

mapped_world_event = world_event_name(selected_disc, selected_sexe)

if mapped_world_event:
    st.info(f"Épreuve mondiale utilisée pour la comparaison : {mapped_world_event}")


# =========================
# Graphique global France
# =========================

st.subheader("Graphique global France")

fig_global = plot_annual_scores(
    df,
    selected_disc,
    selected_sexe,
    selected_cat,
    score_col=score_col
)

if fig_global is not None:
    st.plotly_chart(fig_global, use_container_width=True)
else:
    st.warning("Pas de données pour ce graphique.")

st.divider()


# =========================
# Graphique France vs Monde
# =========================

st.subheader("Comparaison niveau Championnat de France vs niveau mondial")

fig_compare, france_stats, world_stats = plot_france_vs_world(
    df,
    world_df,
    selected_disc,
    selected_sexe,
    selected_cat,
    score_col_france=score_col
)

if fig_compare is not None:
    st.plotly_chart(fig_compare, use_container_width=True)

    with st.expander("Voir les statistiques France"):
        st.dataframe(france_stats, use_container_width=True)

    with st.expander("Voir les statistiques Monde"):
        st.dataframe(world_stats, use_container_width=True)

else:
    st.warning(
        "Impossible d'afficher la comparaison France vs Monde pour cette combinaison. "
        "Vérifie que l'épreuve existe dans le CSV mondial."
    )

st.divider()


# =========================
# Analyse athlète suivi
# =========================

st.subheader("Analyse athlète suivi 2016-2026 avec niveau mondial")

selected_athlete = st.selectbox(
    "Choisir un athlète suivi",
    ATHLETES_SUIVIS
)

fig_athlete_context, athlete_context_stats = plot_athlete_vs_category_by_year(
    df,
    world_df,
    selected_athlete,
    selected_disc,
    score_col=score_col
)

if fig_athlete_context is not None:
    st.plotly_chart(fig_athlete_context, use_container_width=True)

    st.caption(
        "Les lignes France sont calculées chaque année dans la même catégorie réelle que l’athlète. "
        "Les lignes Monde utilisent les résultats de Coupe du monde correspondants à la discipline et au sexe."
    )

    st.dataframe(athlete_context_stats, use_container_width=True)

else:
    st.warning(
        f"Aucune donnée trouvée pour {selected_athlete} en {selected_disc} entre 2016 et 2026."
    )

st.divider()


# =========================
# Analyse athlète classique
# =========================

st.subheader("Analyse athlète classique avec repères mondiaux")

athletes = sorted(filtered_df["athlete"].dropna().unique())

selected_athlete_classic = st.selectbox(
    "Athlète dans les données filtrées",
    athletes
)

summary = athlete_summary(
    df,
    selected_athlete_classic,
    sexe=selected_sexe,
    categorie_age=selected_cat
)

if not summary.empty:
    st.dataframe(summary, use_container_width=True)

score_fig = plot_athlete_score(
    df,
    world_df,
    selected_athlete_classic,
    selected_disc,
    sexe=selected_sexe,
    categorie_age=selected_cat
)

rank_fig = plot_athlete_rank(
    df,
    selected_athlete_classic,
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


# =========================
# Données filtrées
# =========================

with st.expander("Voir les données filtrées France"):
    st.dataframe(filtered_df, use_container_width=True)

st.download_button(
    "Télécharger les données filtrées France en CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="cdf_filtre.csv",
    mime="text/csv",
)

if not world_df.empty:
    with st.expander("Voir les données mondiales"):
        st.dataframe(world_df, use_container_width=True)

    st.download_button(
        "Télécharger les données mondiales nettoyées en CSV",
        data=world_df.to_csv(index=False).encode("utf-8"),
        file_name="world_clean_loaded.csv",
        mime="text/csv",
    )
