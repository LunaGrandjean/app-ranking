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


@st.cache_data(show_spinner=False)
def load_csv_folder(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    files = sorted([
        f for f in os.listdir(path)
        if f.lower().endswith(".csv")
        and f != "all_world_cups_clean.csv"
    ])

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

    df["Total_normalise"] = df["Total"]

    # 10m CDF : normalisation selon le nombre de séries
    mask_10m = (df["discipline"] == "10m") & (df["nb_series"] > 0)
    df.loc[mask_10m, "Total_normalise"] = (
        df.loc[mask_10m, "Total"] * 6 / df.loc[mask_10m, "nb_series"]
    )

    # 50m CDF uniquement : si score > 900, on divise par 2
    mask_50m_cdf_old = (df["discipline"] == "50m") & (df["Total"] > 900)
    df.loc[mask_50m_cdf_old, "Total_normalise"] = (
        df.loc[mask_50m_cdf_old, "Total"] / 2
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


@st.cache_data(show_spinner=False)
def load_world_data(path: str) -> pd.DataFrame:
    file_path = os.path.join(path, "all_world_cups_clean.csv")

    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    df["Rank"] = pd.to_numeric(df.get("rank"), errors="coerce")
    df["Total"] = pd.to_numeric(df.get("qualification"), errors="coerce")
    df["score_finale"] = pd.to_numeric(df.get("final"), errors="coerce")

    if "date_parsed" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
        df["year"] = df["date_parsed"].dt.year
    elif "year_clean" in df.columns:
        df["year"] = pd.to_numeric(df["year_clean"], errors="coerce")
    else:
        df["year"] = pd.to_numeric(df.get("year"), errors="coerce")

    df["athlete"] = df["athlete"].apply(normalize_athlete_name)

    event_col = df["event"].astype(str)

    df["discipline"] = np.where(
        event_col.str.contains("10m", case=False, na=False),
        "10m",
        np.where(
            event_col.str.contains("50m", case=False, na=False),
            "50m",
            "other"
        )
    )

    df["sexe"] = np.where(
        event_col.str.contains("Women", case=False, na=False),
        "Women",
        np.where(
            event_col.str.contains("Men", case=False, na=False),
            "Men",
            "Mixed"
        )
    )

    df["categorie_age"] = "Senior"
    df["competition_level"] = "World Cup"

    # IMPORTANT : World Cup jamais normalisé
    df["Total_normalise"] = df["Total"]

    df = df[
        (df["Rank"].notna())
        & (df["Rank"] > 0)
        & (df["Total"].notna())
        & (df["Total"] > 0)
        & (df["year"].between(2016, 2026))
        & (df["discipline"].isin(["10m", "50m"]))
        & (df["sexe"].isin(["Men", "Women"]))
    ].copy()

    return df


def annual_stats(
    data: pd.DataFrame,
    discipline: str,
    sexe: str,
    categorie_age: str,
    score_col: str = "Total_normalise"
) -> pd.DataFrame:

    sub = data[
        (data["discipline"] == discipline)
        & (data["sexe"] == sexe)
        & (data["categorie_age"] == categorie_age)
    ].copy()

    sub = sub.dropna(subset=[score_col, "year", "competition"])
    sub = sub[sub[score_col] > 0]

    rows = []
    group_cols = ["year", "competition", "Épreuve"]

    for keys, g in sub.groupby(group_cols, dropna=False):
        year = keys[0]

        # On trie par score décroissant, pas par rank final
        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)

        rows.append({
            "year": year,
            "first_score": g.iloc[0][score_col],
            "podium_mean": g.head(3)[score_col].mean(),
            "top10_mean": g.head(10)[score_col].mean(),
            "rank10_score": g.iloc[9][score_col] if len(g) >= 10 else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    yearly = pd.DataFrame(rows).groupby("year", as_index=False).mean(numeric_only=True)
    return yearly.sort_values("year")


def annual_world_stats(
    world_data: pd.DataFrame,
    discipline: str,
    sexe: str,
    score_col: str = "Total_normalise"
) -> pd.DataFrame:

    if world_data.empty:
        return pd.DataFrame()

    sub = world_data[
        (world_data["discipline"] == discipline)
        & (world_data["sexe"] == sexe)
    ].copy()

    sub = sub.dropna(subset=[score_col, "year", "competition", "event"])
    sub = sub[sub[score_col] > 0]

    rows = []

    for keys, g in sub.groupby(["year", "competition", "event"], dropna=False):
        year = keys[0]

        # On trie par score décroissant
        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)

        rows.append({
            "year": year,
            "world_first_score": g.iloc[0][score_col],
            "world_podium_mean": g.head(3)[score_col].mean(),
            "world_top10_mean": g.head(10)[score_col].mean(),
            "world_rank10_score": g.iloc[9][score_col] if len(g) >= 10 else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    yearly = pd.DataFrame(rows).groupby("year", as_index=False).mean(numeric_only=True)
    return yearly.sort_values("year")


def plot_annual_scores(
    df: pd.DataFrame,
    discipline: str,
    sexe: str,
    categorie_age: str,
    score_col: str = "Total_normalise",
    world_data: pd.DataFrame = pd.DataFrame()
):
    stats = annual_stats(df, discipline, sexe, categorie_age, score_col=score_col)

    if stats.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["rank10_score"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["first_score"],
        mode="lines",
        fill="tonexty",
        name="CDF — Couloir 1er-10e",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["first_score"],
        mode="lines+markers",
        name="CDF — 1er score",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["podium_mean"],
        mode="lines+markers",
        name="CDF — Moyenne top 3 scores",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["top10_mean"],
        mode="lines+markers",
        name="CDF — Moyenne top 10 scores",
    ))

    world_stats = annual_world_stats(
        world_data,
        discipline,
        sexe,
        score_col="Total_normalise"
    )

    if not world_stats.empty:
        fig.add_trace(go.Scatter(
            x=world_stats["year"],
            y=world_stats["world_first_score"],
            mode="lines+markers",
            name="World Cup — 1er score",
            line=dict(dash="dash"),
        ))

        fig.add_trace(go.Scatter(
            x=world_stats["year"],
            y=world_stats["world_podium_mean"],
            mode="lines+markers",
            name="World Cup — Moyenne top 3 scores",
            line=dict(dash="dot"),
        ))

        fig.add_trace(go.Scatter(
            x=world_stats["year"],
            y=world_stats["world_top10_mean"],
            mode="lines+markers",
            name="World Cup — Moyenne top 10 scores",
            line=dict(dash="longdash"),
        ))

    fig.update_layout(
        title=f"Évolution annuelle des scores — {discipline} / {sexe} / {categorie_age}",
        xaxis_title="Année",
        yaxis_title="Score",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_athlete_vs_category_by_year(
    data: pd.DataFrame,
    athlete_name: str,
    discipline: str,
    score_col: str = "Total_normalise",
    world_data: pd.DataFrame = pd.DataFrame()
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

        cdf_year_stats = annual_stats(
            data[
                (data["year"] == year)
                & (data["discipline"] == discipline)
                & (data["sexe"] == sexe)
                & (data["categorie_age"] == categorie)
            ],
            discipline,
            sexe,
            categorie,
            score_col=score_col
        )

        cdf_row = cdf_year_stats.iloc[0] if not cdf_year_stats.empty else {}

        rows.append({
            "year": year,
            "athlete_score": athlete_row[score_col],
            "athlete_rank": athlete_row["Rank"],
            "categorie_age": categorie,
            "sexe": sexe,
            "first_score": cdf_row.get("first_score", np.nan),
            "podium_mean": cdf_row.get("podium_mean", np.nan),
            "top10_mean": cdf_row.get("top10_mean", np.nan),
            "rank10_score": cdf_row.get("rank10_score", np.nan),
        })

    stats = pd.DataFrame(rows).sort_values("year")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["rank10_score"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["first_score"],
        mode="lines",
        fill="tonexty",
        name="CDF — Couloir top 1-top 10",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["first_score"],
        mode="lines+markers",
        name="CDF — 1er score catégorie",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["podium_mean"],
        mode="lines+markers",
        name="CDF — Moyenne top 3 catégorie",
    ))

    fig.add_trace(go.Scatter(
        x=stats["year"],
        y=stats["top10_mean"],
        mode="lines+markers",
        name="CDF — Moyenne top 10 catégorie",
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

    if not world_data.empty:
        athlete_sexes = stats["sexe"].dropna().unique()

        if len(athlete_sexes) > 0:
            athlete_sexe = athlete_sexes[0]

            world_stats = annual_world_stats(
                world_data,
                discipline,
                athlete_sexe,
                score_col="Total_normalise"
            )

            if not world_stats.empty:
                fig.add_trace(go.Scatter(
                    x=world_stats["year"],
                    y=world_stats["world_first_score"],
                    mode="lines+markers",
                    name="World Cup — 1er score",
                    line=dict(dash="dash"),
                ))

                fig.add_trace(go.Scatter(
                    x=world_stats["year"],
                    y=world_stats["world_podium_mean"],
                    mode="lines+markers",
                    name="World Cup — Moyenne top 3",
                    line=dict(dash="dot"),
                ))

                fig.add_trace(go.Scatter(
                    x=world_stats["year"],
                    y=world_stats["world_top10_mean"],
                    mode="lines+markers",
                    name="World Cup — Moyenne top 10",
                    line=dict(dash="longdash"),
                ))

    fig.update_layout(
        title=f"Évolution 2016-2026 — {athlete_name} vs CDF et niveau mondial",
        xaxis_title="Année",
        yaxis_title="Score",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig, stats


def plot_athlete_score(
    data: pd.DataFrame,
    athlete_name: str,
    discipline: str,
    sexe: Optional[str] = None,
    categorie_age: Optional[str] = None
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
        y="Total_normalise",
        color="categorie_age",
        markers=True,
        hover_data=["Rank", "discipline", "sexe", "source_pdf", "Total"],
        title=f"Évolution du score — {athlete_name} ({discipline})",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Année",
        yaxis_title="Score"
    )

    return fig


def plot_athlete_rank(
    data: pd.DataFrame,
    athlete_name: str,
    discipline: str,
    sexe: Optional[str] = None,
    categorie_age: Optional[str] = None
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
        hover_data=["Total", "Total_normalise", "discipline", "sexe", "source_pdf"],
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
    categorie_age: Optional[str] = None
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
        [
            "about", "year", "discipline", "sexe", "categorie_age",
            "Rank", "Total", "Total_normalise", "source_pdf"
        ]
    ].sort_values(["discipline", "year"])


# =========================
# Chargement auto
# =========================

raw = load_csv_folder("csv")
df = prepare_data(raw) if not raw.empty else pd.DataFrame()
world_df = load_world_data("csv")

st.title("Analyse CDF — version Streamlit")
st.caption("Analyse automatique des CSV du dossier csv/ avec comparaison World Cup")

if raw.empty:
    st.error("Aucun fichier CSV CDF trouvé dans le dossier 'csv/'.")
    st.stop()

if df.empty:
    st.warning("Les fichiers CDF ont été chargés, mais aucune ligne exploitable n'a été trouvée après nettoyage.")
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Lignes brutes CDF", len(raw))
col2.metric("Lignes analysables CDF", len(df))
col3.metric("Athlètes CDF", df["athlete"].nunique())
col4.metric("Années CDF", int(df["year"].nunique()) if df["year"].notna().any() else 0)
col5.metric("Lignes World Cup", len(world_df))

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
    "Catégorie",
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

score_col = "Total_normalise"

# =========================
# Graphique global
# =========================

st.subheader("Graphique global CDF vs World Cup")

fig_global = plot_annual_scores(
    df,
    selected_disc,
    selected_sexe,
    selected_cat,
    score_col=score_col,
    world_data=world_df
)

if fig_global is not None:
    st.plotly_chart(fig_global, use_container_width=True)
else:
    st.warning("Pas de données pour ce graphique.")

st.divider()

# =========================
# Analyse athlète suivie
# =========================

st.subheader("Analyse athlète suivi 2016-2026")

selected_athlete = st.selectbox(
    "Choisir un athlète",
    ATHLETES_SUIVIS
)

fig_athlete_context, athlete_context_stats = plot_athlete_vs_category_by_year(
    df,
    selected_athlete,
    selected_disc,
    score_col=score_col,
    world_data=world_df
)

if fig_athlete_context is not None:
    st.plotly_chart(fig_athlete_context, use_container_width=True)

    st.caption(
        "Les lignes CDF / World Cup sont calculées sur les meilleurs scores, pas uniquement sur le Rank final."
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

st.subheader("Analyse athlète classique")

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

with st.expander("Voir les données CDF filtrées"):
    st.dataframe(filtered_df, use_container_width=True)

if not world_df.empty:
    world_filtered = world_df[
        (world_df["sexe"] == selected_sexe)
        & (world_df["discipline"] == selected_disc)
    ].copy()

    with st.expander("Voir les données World Cup filtrées"):
        st.dataframe(world_filtered, use_container_width=True)

st.download_button(
    "Télécharger les données CDF filtrées en CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="cdf_filtre.csv",
    mime="text/csv",
)
