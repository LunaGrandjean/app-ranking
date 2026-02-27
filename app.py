import re
import io
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

EXCEL_ENGINE = "openpyxl"

# =========================
# Utils: date conversion
# =========================
def excel_date_to_date(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, float, np.integer, np.floating)):
        return (pd.Timestamp("1899-12-30") + pd.to_timedelta(int(x), unit="D")).date()
    if isinstance(x, pd.Timestamp):
        return x.date()

    s = str(x).strip()
    if not s:
        return pd.NaT

    # essaie d'abord ISO (silence le warning)
    dtv = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    if pd.isna(dtv):
        # fallback dayfirst
        dtv = pd.to_datetime(s, errors="coerce", dayfirst=True)

    return dtv.date() if not pd.isna(dtv) else pd.NaT

def squish(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

# =========================
# 1) Extract workbook (raw FFTir Excel -> long df)
# =========================
def extract_sheet_long(raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    raw = raw.copy()

    # --- 1) Trouver la ligne header ("Cat" + "Name") ---
    header_row = None
    for r in range(min(40, raw.shape[0])):  # on cherche dans le haut du fichier
        c0 = squish(raw.iat[r, 0]).lower()
        c1 = squish(raw.iat[r, 1]).lower()
        if c0 == "cat" and c1 == "name":
            header_row = r
            break

    if header_row is None:
        # fallback si jamais ça bouge : renvoie vide (et tu verras dans "Données clean")
        return pd.DataFrame(columns=["Athlete","Category","Competition","Date","Score","Rank","Sheet"])

    athlete_start_row = header_row + 1
    comp_row = max(0, header_row - 2)  # comme ton fichier: compet est 2 lignes au-dessus

    # --- 2) Trouver la 1ère colonne "Score" des compétitions sur la ligne header ---
    score_cols = []
    for c in range(raw.shape[1]):
        if squish(raw.iat[header_row, c]).lower() == "score":
            score_cols.append(c)

    if not score_cols:
        return pd.DataFrame(columns=["Athlete","Category","Competition","Date","Score","Rank","Sheet"])

    first_comp_col = score_cols[0]

    # Step: distance entre 2 colonnes "Score" (souvent 3)
    step = (score_cols[1] - score_cols[0]) if len(score_cols) >= 2 else 3

    # --- Athlètes / Catégories ---
    athletes = raw.iloc[athlete_start_row:, 1].apply(squish).replace({"": np.nan})
    categories = raw.iloc[athlete_start_row:, 0].apply(squish).replace({"": np.nan})

    parts = []
    for col in range(first_comp_col, raw.shape[1] - 1, step):
        comp_name = squish(raw.iat[comp_row, col])
        if comp_name == "":
            continue

        comp_date_raw = raw.iat[comp_row, col + 1] if col + 1 < raw.shape[1] else pd.NaT
        comp_date = excel_date_to_date(comp_date_raw)

        score = pd.to_numeric(raw.iloc[athlete_start_row:, col], errors="coerce")
        rank  = pd.to_numeric(raw.iloc[athlete_start_row:, col + 1], errors="coerce")

        df = pd.DataFrame({
            "Athlete": athletes,
            "Category": categories,
            "Competition": comp_name,
            "Date": comp_date,
            "Score": score,
            "Rank": rank,
            "Sheet": sheet_name
        })

        df = df[df["Athlete"].notna()]                 # athlete non vide
        df = df[~(df["Score"].isna() & df["Rank"].isna())]  # comme R: si score & rank NA -> skip

        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["Athlete","Category","Competition","Date","Score","Rank","Sheet"])

    return pd.concat(parts, ignore_index=True) out

def extract_workbook(file_bytes: bytes) -> pd.DataFrame:
    # ExcelFile + engine forcé
    xf = pd.ExcelFile(io.BytesIO(file_bytes), engine=EXCEL_ENGINE)

    dfs = []
    for sheet in xf.sheet_names:
        try:
            # IMPORTANT: recréer BytesIO à chaque read_excel (évite les soucis de curseur)
            raw = pd.read_excel(io.BytesIO(file_bytes),
                                sheet_name=sheet,
                                header=None,
                                engine=EXCEL_ENGINE,
                                dtype=object)
            dfs.append(extract_sheet_long(raw, sheet))
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame(columns=["Athlete","Category","Competition","Date","Score","Rank","Sheet"])
    return pd.concat(dfs, ignore_index=True)

# =========================
# 2) Competition coefficient (as in R)
# =========================
def add_comp_coeff(df: pd.DataFrame, coeff_table: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # clean table
    ct = coeff_table.copy()
    ct.columns = [c.strip().upper() for c in ct.columns]

    if "MATCH" not in ct.columns or "COEFF" not in ct.columns:
        raise ValueError("La table coeff doit contenir les colonnes MATCH et COEFF")

    ct["MATCH"] = ct["MATCH"].astype(str).str.strip()
    ct["COEFF"] = pd.to_numeric(ct["COEFF"], errors="coerce")
    ct = ct.dropna(subset=["MATCH", "COEFF"])

    # build regex rules (word boundary like R \\b)
    rules = [(rf"\b{re.escape(m)}\b", float(v)) for m, v in zip(ct["MATCH"], ct["COEFF"])]

    comp = df["Competition"].fillna("").astype(str)

    def coeff_one(s: str) -> float:
        for pat, val in rules:
            if re.search(pat, s):
                return val
        return np.nan

    df["Coeff"] = comp.apply(coeff_one)
    return df


# =========================
# 3) Name fixes 
# =========================
def fix_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    repl = {
        r"^BRIAN Julie$": "BRIAND Julie",
    }
    df["Athlete"] = df["Athlete"].replace(repl, regex=True)
    df["Athlete"] = df["Athlete"].replace({
        "BRACKMAN Anceline": "BRACKMAN Ancéline",
        "CADET Madisson": "CADET Madison",
        "BLATEAUX Félix": "BLATEAU Félix",
        "VEDEL Thimothée": "VEDEL Timothée",
        "TENERIFE Alaisia": "TENERIFE Alaisa",
        "BONNET Marcellin": "BONNET Marcelin",
        "CHOLET Mathis": "CHOLET Matis",
    })
    # cas "IUNG Antoine*" (regex)
    df["Athlete"] = df["Athlete"].str.replace(r"IUNG Antoine\*+", "IUNG Antoine", regex=True)
    return df


# =========================
# 4) Parse "point scale" table (your screenshot)
#    Expected blocks: "50m Women", "50m Men", "50m Junior Women", "50m Junior Men"
#                     "10m Women", "10m Men", "10m Junior Women", "10m Junior Men"
#    Each block = two columns: threshold_score, points
# =========================
def parse_scale_from_excel(file_bytes: bytes, sheet_name: str = None) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    xf = pd.ExcelFile(bio, engine=EXCEL_ENGINE)
    use_sheet = sheet_name or xf.sheet_names[0]
    raw = pd.read_excel(bio, sheet_name=use_sheet, header=None, engine=EXCEL_ENGINE)

    # Find headers like "50m Women" in the grid
    headers = {}
    for r in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            v = raw.iat[r, c]
            if isinstance(v, str):
                t = v.strip()
                if t in {"50m Women","50m Men","50m Junior Women","50m Junior Men",
                         "10m Women","10m Men","10m Junior Women","10m Junior Men"}:
                    headers[t] = (r, c)

    rows = []
    for title, (r0, c0) in headers.items():
        # data starts next row; two columns: score at c0, points at c0+1
        rr = r0 + 1
        while rr < raw.shape[0]:
            score = raw.iat[rr, c0]
            pts = raw.iat[rr, c0 + 1] if c0 + 1 < raw.shape[1] else np.nan
            if pd.isna(score) and pd.isna(pts):
                break
            score_num = pd.to_numeric(score, errors="coerce")
            pts_num = pd.to_numeric(pts, errors="coerce")
            if not pd.isna(score_num) and not pd.isna(pts_num):
                rows.append({"ScaleKey": title, "MinScore": float(score_num), "Points": float(pts_num)})
            rr += 1

    scale = pd.DataFrame(rows)
    if scale.empty:
        return scale

    # Ensure descending by MinScore for easy lookup (>=)
    scale = scale.sort_values(["ScaleKey","MinScore"], ascending=[True, False]).reset_index(drop=True)
    return scale


# =========================
# 5) Apply scale: score -> points (step function)
#    Rule: points for the highest MinScore <= score
# =========================
def score_to_points(score: float, scale_df: pd.DataFrame, key: str) -> float:
    if pd.isna(score):
        return np.nan
    sub = scale_df[scale_df["ScaleKey"] == key]
    if sub.empty:
        return np.nan
    # find first row where score >= MinScore (since MinScore sorted desc)
    hit = sub[sub["MinScore"] <= score]
    if hit.empty:
        # below lowest threshold -> take last row's points (typically 0 or 4/5 depending your table)
        return float(sub.iloc[-1]["Points"])
    return float(hit.iloc[0]["Points"])


def derive_sex_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convertit Sheet en texte sans utiliser le dtype "string"
    sheet = df["Sheet"].fillna("").astype(str)

    df["Sexe"] = np.where(
        sheet.str.contains("DAMES", case=False, na=False),
        "F",
        "M"
    )

    # Distance en object (pas np.nan float)
    df["Distance"] = None
    df.loc[sheet.str.contains("10m", case=False, na=False), "Distance"] = "10m"
    df.loc[sheet.str.contains("50m", case=False, na=False), "Distance"] = "50m"

    return df

def scale_key(distance: str, sexe: str, category: str) -> str:
    # category in your file seems "J" or "S"
    # Junior => "Junior", Senior => no "Junior"
    if pd.isna(distance) or pd.isna(sexe):
        return None
    is_junior = (str(category).strip().upper() == "J")
    women = (str(sexe).upper() == "F")
    if distance == "50m":
        if is_junior and women: return "50m Junior Women"
        if is_junior and not women: return "50m Junior Men"
        if (not is_junior) and women: return "50m Women"
        return "50m Men"
    if distance == "10m":
        if is_junior and women: return "10m Junior Women"
        if is_junior and not women: return "10m Junior Men"
        if (not is_junior) and women: return "10m Women"
        return "10m Men"
    return None


# =========================
# 6) Final points mapping (Rank -> FinalPoints)
# =========================
def default_final_points():
    # You can edit this in the app
    return pd.DataFrame({
        "Rank": [1,2,3,4,5,6,7,8],
        "FinalPoints": [15,14,13,9,8,7,6,5]
    })


# =========================
# 7) Time coefficient (same shape as R)
# =========================
def add_date_coeff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    today = dt.date.today()

    diff_months = df["Date"].apply(lambda d: np.nan if pd.isna(d) else (today.year - d.year) * 12 + (today.month - d.month))
    # approximate months diff like R interval/months
    k = 3.0

    def f(m):
        if pd.isna(m): return np.nan
        if m <= 4: return 1.0
        if m >= 12: return 0.0
        return (1 - np.exp(-k * (12 - m) / (12 - 4))) / (1 - np.exp(-k))

    df["date_coeff"] = diff_months.apply(f)
    return df


# =========================
# 8) Final table (Athlete, Distance) like your R summarise
# =========================
def make_final_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    date_ref = df["Date"].dropna().max()

    def n_comp_12m(g):
        if pd.isna(date_ref):
            return 0
        return int(((~g["Score"].isna()) & (g["Date"] >= (date_ref - dt.timedelta(days=365)))).sum())

    def avg_without_worst_two(g):
        s = g["Score"].dropna().sort_values()
        if len(s) <= 2:
            return np.nan
        return float(s.iloc[2:].mean())

    def av_12m(g):
        if pd.isna(date_ref):
            return np.nan
        s = g.loc[(~g["Score"].isna()) & (g["Date"] >= (date_ref - dt.timedelta(days=365))), "Score"]
        return float(s.mean()) if len(s) else np.nan

    def hp_12m(g):
        if pd.isna(date_ref):
            return np.nan
        s = g.loc[(~g["Score"].isna()) & (g["Date"] >= (date_ref - dt.timedelta(days=365))), "Score"]
        return float(s.max()) if len(s) else np.nan

    def weighted_mean(vals, w):
        vals = np.array(vals, dtype=float)
        w = np.array(w, dtype=float)
        mask = ~np.isnan(vals) & ~np.isnan(w)
        if mask.sum() == 0:
            return np.nan
        denom = w[mask].sum()
        if denom == 0:
            return np.nan
        return float((vals[mask] * w[mask]).sum() / denom)

    rows = []
    for (ath, dist), g in df.groupby(["Athlete","Distance"], dropna=False):
        g = g.copy()
        n12 = n_comp_12m(g)

        pctJ_t = weighted_mean(g["%J"], g["Coeff"] * g["date_coeff"])
        pctS_t = weighted_mean(g["%S"], g["Coeff"] * g["date_coeff"])
        pctJ   = weighted_mean(g["%J"], g["Coeff"])
        pctS   = weighted_mean(g["%S"], g["Coeff"])

        row = {
            "Athlete": ath,
            "Distance": dist,
            "n_compet_12m": n12,
            "average_without_worst_two": avg_without_worst_two(g),
            "AV": av_12m(g),
            "HP": hp_12m(g),
            "%J_Temps": pctJ_t,
            "%S_Temps": pctS_t,
            "%J": pctJ,
            "%S": pctS,
            "Sexe": g["Sexe"].dropna().iloc[0] if g["Sexe"].notna().any() else np.nan,
            "Category": g["Category"].dropna().iloc[0] if g["Category"].notna().any() else np.nan,
        }

        # rules: min comps
        if n12 < 4:
            for k in ["%J_Temps","%S_Temps","%J","%S","AV","HP"]:
                row[k] = 0
        if n12 < 5:
            row["average_without_worst_two"] = 0

        # rule seniors: %J and %J_Temps NA if Category == "S"
        if str(row["Category"]).strip().upper() == "S":
            row["%J"] = np.nan
            row["%J_Temps"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Ranking App", layout="wide")
st.title("📊 Ranking App (Excel → paramètres → tableau final)")

colA, colB = st.columns(2)

with colA:
    data_file = st.file_uploader("📥 Fichier résultats", type=["xlsx"], accept_multiple_files=False)

with colB:
    scale_file = st.file_uploader("📥 Point scale", type=["xlsx"], accept_multiple_files=False)

st.divider()

# Final points: upload optional (csv/xlsx) or editable table
st.subheader("Points de finale (Rank → Points)")
fp_upload = st.file_uploader("Optionnel: importer une table points finale (csv/xlsx)", type=["csv","xlsx"], accept_multiple_files=False)

if fp_upload is None:
    final_points_df = default_final_points()
else:
    ext = fp_upload.name.split(".")[-1].lower()
    if ext == "csv":
        final_points_df = pd.read_csv(fp_upload)
    else:
        final_points_df = pd.read_excel(fp_upload, engine=EXCEL_ENGINE)

final_points_df = st.data_editor(final_points_df, num_rows="dynamic", use_container_width=True)

st.divider()

st.subheader("Coeff compétitions (match → coeff)")

def default_coeff_table():
    return pd.DataFrame({
        "MATCH": ["EN", "LOC", "GP", "INT", "CDF", "JECH", "ECH", "JWCH", "WCH", "JWC", "WC", "WCF"],
        "COEFF": [2,   0.5,   3,    2,     1,     3,     3,     5,     5,    4,   4,   5]
    })

coeff_upload = st.file_uploader(
    "Optionnel : importer une table coeff (csv/xlsx)",
    type=["csv", "xlsx"],
    accept_multiple_files=False
)

if coeff_upload is None:
    coeff_df = default_coeff_table()
else:
    ext = coeff_upload.name.split(".")[-1].lower()
    coeff_df = pd.read_csv(coeff_upload) if ext == "csv" else pd.read_excel(coeff_upload, engine=EXCEL_ENGINE)

coeff_df = st.data_editor(coeff_df, num_rows="dynamic", use_container_width=True)

# Scale: either parse from uploaded excel or provide editable default skeleton
st.subheader("Barème (Point scale)")
if scale_file is not None:
    scale_df = parse_scale_from_excel(scale_file.getvalue())
    if scale_df.empty:
        st.warning("Je n’ai pas trouvé les blocs (ex: '50m Women', '10m Junior Men', etc.).")
else:
    # empty template user can paste
    scale_df = pd.DataFrame({"ScaleKey": [], "MinScore": [], "Points": []})

scale_df = st.data_editor(scale_df, num_rows="dynamic", use_container_width=True)

st.divider()

# Athletes to remove
st.subheader("Athlètes à exclure")
remove_text = st.text_area("1 nom par ligne", value="BAILLY Nathan\nFORET Alexandre\nGERMOND Etienne\nMOMBERT Claire", height=110)
athletes_to_remove = [x.strip() for x in remove_text.splitlines() if x.strip()]

run = st.button("Calculer le tableau final", type="primary")

if run:
    if data_file is None:
        st.error("Upload le fichier Excel résultats (FFTir) d’abord.")
        st.stop()

    if scale_df.empty:
        st.error("Il me faut un barème (point scale). Upload le fichier barème ou remplis la table dans l’app.")
        st.stop()

    # Validate scale columns
    needed = {"ScaleKey","MinScore","Points"}
    if not needed.issubset(scale_df.columns):
        st.error("Le barème doit contenir les colonnes: ScaleKey, MinScore, Points")
        st.stop()

    # Make sure numeric
    scale_df = scale_df.copy()
    scale_df["MinScore"] = pd.to_numeric(scale_df["MinScore"], errors="coerce")
    scale_df["Points"] = pd.to_numeric(scale_df["Points"], errors="coerce")
    scale_df = scale_df.dropna(subset=["ScaleKey","MinScore","Points"])
    scale_df = scale_df.sort_values(["ScaleKey","MinScore"], ascending=[True, False]).reset_index(drop=True)

    # final points numeric
    fp = final_points_df.copy()
    fp["Rank"] = pd.to_numeric(fp["Rank"], errors="coerce")
    fp["FinalPoints"] = pd.to_numeric(fp["FinalPoints"], errors="coerce")
    fp = fp.dropna(subset=["Rank","FinalPoints"])
    fp_map = dict(zip(fp["Rank"].astype(int), fp["FinalPoints"].astype(float)))

    # Extract data
    df = extract_workbook(data_file.getvalue())

    # Clean like your script
    df = df[df["Score"].fillna(0) > 0].copy()
    df = df[df["Score"].notna()].copy()

    # Add Sexe/Distance + comp coeff + names
    df = derive_sex_distance(df)
    df = add_comp_coeff(df, coeff_df)
    df = fix_names(df)

    # Remove athletes
    if athletes_to_remove:
        df = df[~df["Athlete"].isin(athletes_to_remove)].copy()

    # Date coeff
    df = add_date_coeff(df)

    # Perf via scale table:
    # We compute Perf_S and Perf_J based on Category (J/S) using the right ScaleKey
    # In practice: for each row, choose key from (Distance, Sexe, Category)
    df["ScaleKey"] = df.apply(lambda r: scale_key(r["Distance"], r["Sexe"], r["Category"]), axis=1)

    # Apply points (Perf) from scale
    df["Perf"] = df.apply(lambda r: score_to_points(r["Score"], scale_df, r["ScaleKey"]), axis=1)

    # To keep your same columns (%J and %S), we mirror:
    # Perf_S = senior perf table, Perf_J = junior perf table
    # Here we can compute both by forcing Category in key.
    df["ScaleKey_S"] = df.apply(lambda r: scale_key(r["Distance"], r["Sexe"], "S"), axis=1)
    df["ScaleKey_J"] = df.apply(lambda r: scale_key(r["Distance"], r["Sexe"], "J"), axis=1)
    df["Perf_S"] = df.apply(lambda r: score_to_points(r["Score"], scale_df, r["ScaleKey_S"]), axis=1)
    df["Perf_J"] = df.apply(lambda r: score_to_points(r["Score"], scale_df, r["ScaleKey_J"]), axis=1)

    # Final score from rank mapping
    def rank_to_finalpts(x):
        if pd.isna(x):
            return np.nan
        try:
            return fp_map.get(int(x), np.nan)
        except Exception:
            return np.nan

    df["Finale_score"] = df["Rank"].apply(rank_to_finalpts)

    df["Total_Score_J"] = df["Perf_J"] + np.where(df["Finale_score"].isna(), 0, df["Finale_score"])
    df["Total_Score_S"] = df["Perf_S"] + np.where(df["Finale_score"].isna(), 0, df["Finale_score"])
    df["%J"] = (df["Total_Score_J"] / 50.0) * 100.0
    df["%S"] = (df["Total_Score_S"] / 50.0) * 100.0

    # Filter distance known
    df = df[df["Distance"].notna()].copy()

    # Final table
    final_tbl = make_final_table(df)

    # Display
    tab1, tab2 = st.tabs(["Données clean (long)", "Tableau final"])
    with tab1:
        st.dataframe(df, use_container_width=True, height=520)
    with tab2:
        st.dataframe(final_tbl, use_container_width=True, height=520)

    # Export
    st.subheader("Exports")
    c1, c2 = st.columns(2)

    with c1:
        csv_bytes = final_tbl.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger tableau_final.csv", data=csv_bytes, file_name="tableau_final.csv", mime="text/csv")

    with c2:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            final_tbl.to_excel(writer, index=False, sheet_name="Final")
            df.to_excel(writer, index=False, sheet_name="Clean_Long")
            scale_df.to_excel(writer, index=False, sheet_name="Scale")
            fp.to_excel(writer, index=False, sheet_name="FinalPoints")
        st.download_button(
            "Télécharger tableau_final.xlsx",
            data=out.getvalue(),
            file_name="tableau_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



        




