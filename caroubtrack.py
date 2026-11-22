#!/usr/bin/env python3
"""
CaroubTrack – Phénotypage morphométrique du caroubier (Ceratonia siliqua L.)
Streamlit + SQLite + pandas/numpy/plotly (+ OpenCV, SciPy optionnels)

Installation :  pip install streamlit pandas numpy plotly scipy opencv-python-headless
Lancement    :  streamlit run caroubtrack.py

Fichier unique (anciennement caroubtrack.py + caroub_extras.py).
Modules : arbres/accessions · saisie en tableau (gousses, feuilles/folioles, graines)
          photo-mesure calibrée par pièce · indices dérivés · stats par cultivar
          corrélations · ACP · classification hiérarchique · import/export CSV
"""
import sqlite3, datetime, math
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from types import SimpleNamespace
import itertools

try:
    import cv2; CV2 = True
except ImportError:
    CV2 = False
try:
    from scipy import stats
    SCIPY = True
except ImportError:
    SCIPY = False

st.set_page_config(page_title="CaroubTrack", page_icon="🌿", layout="wide")
DB = "caroub.db"

# Exemples de cultivars méditerranéens : à adapter à votre collection
CULTIVARS = ["Sfax", "Tylliria", "Mulata", "Santa Fe", "Duraió", "Rojal",
             "Banya de Cabra", "Négra", "Gasparet", "Sauvage (silvestris)", "Local / inconnu"]
SEXES = ["femelle", "mâle", "hermaphrodite", "non déterminé"]

# Descripteurs mesurés (colonne SQL, libellé)
COLOR = [("L_star", "L* (clarté CIELAB)"), ("a_star", "a* (vert−/rouge+)"), ("b_star", "b* (bleu−/jaune+)")]
TRAITS = {
    "gousses": ("🫘 Gousses", [
        ("longueur_mm", "Longueur (mm)"), ("largeur_mm", "Largeur (mm)"),
        ("epaisseur_mm", "Épaisseur (mm)"), ("poids_g", "Poids (g)"),
        ("nb_graines", "Nb graines"), ("poids_graines_g", "Poids graines (g)"),
        ("sucres_brix", "Sucres pulpe (°Brix)")] + COLOR),
    "feuilles": ("🍃 Feuilles / folioles", [
        ("nb_paires_folioles", "Paires de folioles"), ("long_foliole_mm", "Long. foliole (mm)"),
        ("larg_foliole_mm", "Larg. foliole (mm)"), ("long_rachis_mm", "Long. rachis (mm)"),
        ("masse_seche_foliole_mg", "Masse sèche foliole (mg)"), ("spad", "Indice chlorophylle SPAD")] + COLOR),
    "graines": ("🌰 Graines", [
        ("longueur_mm", "Longueur (mm)"), ("largeur_mm", "Largeur (mm)"),
        ("epaisseur_mm", "Épaisseur (mm)"), ("poids_mg", "Poids (mg)")] + COLOR),
    "architecture": ("🌳 Architecture arbre", [
        ("hauteur_m", "Hauteur totale (m)"), ("haut_tronc_m", "Hauteur du tronc libre (m)"),
        ("diam_cime_1_m", "Diamètre cime 1 (m)"), ("diam_cime_2_m", "Diamètre cime 2 (m)"),
        ("circ_tronc_cm", "Circonférence tronc à 1,3 m (cm)")]),
    "biochimie": ("🧪 Biochimie", [
        ("sucres_totaux_pct", "Sucres totaux pulpe (% MS)"), ("saccharose_pct", "Saccharose (% MS)"),
        ("glucose_pct", "Glucose (% MS)"), ("fructose_pct", "Fructose (% MS)"),
        ("tanins_mg_eag_g", "Tanins (mg EAG/g MS)"), ("polyphenols_mg_eag_g", "Polyphénols (mg EAG/g MS)"),
        ("masse_graines_g", "Masse graines (g)"), ("masse_endosperme_g", "Masse endosperme (g)"),
        ("masse_gomme_g", "Masse gomme LBG extraite (g)"), ("galactomannane_pct", "Galactomannane (% gomme)")]),
}
PREFIX = {"gousses": "g", "feuilles": "f", "graines": "s", "biochimie": "b", "architecture": "a"}


# ─────────────────────────── Base de données ───────────────────────────
def db():
    c = sqlite3.connect(DB)
    c.execute("PRAGMA foreign_keys=ON")
    return c

def q(sql, params=()):
    with db() as c:
        return pd.read_sql(sql, c, params=params)

def ex(sql, params=()):
    with db() as c:
        c.execute(sql, params)

def init_db():
    ex("""CREATE TABLE IF NOT EXISTS arbres(
        id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT UNIQUE NOT NULL, cultivar TEXT,
        sexe TEXT, latitude REAL, longitude REAL, altitude_m REAL, age_ans INTEGER, notes TEXT)""")
    for t, (_, cols) in TRAITS.items():
        ex(f"""CREATE TABLE IF NOT EXISTS {t}(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arbre_id INTEGER REFERENCES arbres(id) ON DELETE CASCADE,
            date_mesure TEXT, {', '.join(c + ' REAL' for c, _ in cols)}, source TEXT)""")

def migrate():
    for t, (_, cols) in TRAITS.items():
        have = set(q(f"PRAGMA table_info({t})").name)
        for c, _ in cols:
            if c not in have:
                ex(f"ALTER TABLE {t} ADD COLUMN {c} REAL")

def ins(t, aid, vals, source="manuel"):
    cols = [c for c, _ in TRAITS[t][1]]
    ex(f"INSERT INTO {t}(arbre_id,date_mesure,{','.join(cols)},source) VALUES(?,?,{','.join('?' * len(cols))},?)",
       [aid, str(datetime.date.today())] + [None if pd.isna(vals.get(c)) else float(vals[c]) for c in cols] + [source])


# ─────────────────────────── Indices dérivés ───────────────────────────
def add_indices(t, d):
    d = d.copy()
    if d.empty:
        return d
    for c, _ in TRAITS[t][1]:  # colonnes vides (NULL) → numériques
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if t == "gousses":
        d["ratio_L_l"] = d.longueur_mm / d.largeur_mm
        d["pulpe_pct"] = (d.poids_g - d.poids_graines_g) / d.poids_g * 100
        d["poids_moy_graine_mg"] = d.poids_graines_g / d.nb_graines.replace(0, np.nan) * 1000
    elif t == "feuilles":
        d["ratio_L_l"] = d.long_foliole_mm / d.larg_foliole_mm
        d["surface_foliole_cm2"] = math.pi / 4 * d.long_foliole_mm * d.larg_foliole_mm / 100  # ellipse
        d["sla_cm2_g"] = d.surface_foliole_cm2 / (d.masse_seche_foliole_mg / 1000)
    elif t == "architecture":
        d["surface_projection_m2"] = math.pi / 4 * d.diam_cime_1_m * d.diam_cime_2_m
        d["volume_cime_m3"] = math.pi / 6 * (d.hauteur_m - d.haut_tronc_m) * d.diam_cime_1_m * d.diam_cime_2_m
        d["diam_tronc_cm"] = d.circ_tronc_cm / math.pi
    elif t == "biochimie":
        d["endosperme_pct_graine"] = d.masse_endosperme_g / d.masse_graines_g * 100
        d["rendement_gomme_pct_graine"] = d.masse_gomme_g / d.masse_graines_g * 100
        d["rendement_gomme_pct_endosperme"] = d.masse_gomme_g / d.masse_endosperme_g * 100
    else:
        d["ratio_L_l"] = d.longueur_mm / d.largeur_mm
        d["sphericite_pct"] = (d.longueur_mm * d.largeur_mm * d.epaisseur_mm) ** (1 / 3) / d.longueur_mm * 100
    if "a_star" in d:  # colorimétrie : chroma C* et angle de teinte h°
        d["chroma_C"] = np.hypot(d.a_star, d.b_star)
        d["teinte_h_deg"] = np.degrees(np.arctan2(d.b_star, d.a_star)) % 360
    return d.replace([np.inf, -np.inf], np.nan)

def load(t):
    d = q(f"SELECT m.*, a.code, a.cultivar FROM {t} m JOIN arbres a ON a.id=m.arbre_id")
    return add_indices(t, d)

def wide():
    """Moyennes par arbre de tous les descripteurs (matrice pour ACP / classification)."""
    out = None
    for t, p in PREFIX.items():
        d = load(t)
        if d.empty:
            continue
        num = d.select_dtypes("number").drop(columns=["id", "arbre_id"])
        m = num.groupby(d.arbre_id).mean().add_prefix(p + "_")
        out = m if out is None else out.join(m, how="outer")
    if out is None:
        return pd.DataFrame()
    return q("SELECT id,code,cultivar FROM arbres").set_index("id").join(out, how="inner")


# ─────────────────────────── Photo-mesure (OpenCV) ───────────────────────────
def measure_photo(data, coin_mm, min_mm2):
    """Pièce étalon (cercle de Hough) → échelle mm/px → contours des objets sombres sur fond clair."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    s = 1400 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s)
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=25,
                         minRadius=15, maxRadius=int(min(img.shape[:2]) / 4))
    if c is None:
        return [], img, "Pièce étalon non détectée (fond uni, bon éclairage, pièce bien visible)."
    x, y, r = c[0][0]
    px_mm = 2 * r / coin_mm
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.circle(th, (int(x), int(y)), int(r * 1.2), 0, -1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    for k in cnts:
        area = cv2.contourArea(k) / px_mm ** 2
        if area < min_mm2:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(k)
        L, W = max(w, h) / px_mm, min(w, h) / px_mm
        Lc = max(cv2.arcLength(k, True) / px_mm / 2 - W, L)  # longueur curviligne approx. (gousses courbes)
        objs.append(dict(L=L, W=W, Lcurv=Lc, aire_mm2=area, k=k))
    cv2.circle(img, (int(x), int(y)), int(r), (0, 200, 0), 3)
    for i, o in enumerate(objs, 1):
        cv2.drawContours(img, [o["k"]], -1, (0, 0, 255), 2)
        bx, by, _, _ = cv2.boundingRect(o["k"])
        cv2.putText(img, str(i), (bx, max(by - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return objs, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"Échelle : {px_mm:.2f} px/mm · {len(objs)} objet(s)"


# ─────────────────────────── Pages ───────────────────────────
def arbre_opts():
    a = q("SELECT id, code, cultivar FROM arbres ORDER BY code")
    return {f"{r.code} ({r.cultivar})": int(r.id) for r in a.itertuples()}

def demo():
    rng = np.random.default_rng(1)
    for ci, cv in enumerate(CULTIVARS[:4]):
        for n in range(5):
            code = f"DEMO-{cv[:3].upper()}-{n + 1}"
            ex("INSERT OR IGNORE INTO arbres(code,cultivar,sexe,latitude,longitude) VALUES(?,?,?,?,?)",
               (code, cv, "femelle", 35 + rng.normal(0, .02), 2 + rng.normal(0, .02)))
            aid = int(q("SELECT id FROM arbres WHERE code=?", (code,)).id[0])
            k = (0.85 + 0.1 * ci) * rng.normal(1, .05)
            for _ in range(10):
                ng = max(int(rng.normal(10 * k, 1.5)), 1)
                ins("gousses", aid, dict(longueur_mm=rng.normal(160 * k, 12), largeur_mm=rng.normal(19 * k, 1.5),
                    epaisseur_mm=rng.normal(8 * k, .7), poids_g=rng.normal(16 * k, 2.5), nb_graines=ng,
                    poids_graines_g=ng * rng.normal(.19, .015), sucres_brix=rng.normal(45, 4)), "démo simulée")
                ins("feuilles", aid, dict(nb_paires_folioles=rng.integers(3, 7), long_foliole_mm=rng.normal(55 * k, 5),
                    larg_foliole_mm=rng.normal(38 * k, 4), long_rachis_mm=rng.normal(110 * k, 12)), "démo simulée")
                ins("graines", aid, dict(longueur_mm=rng.normal(8.5 * k, .5), largeur_mm=rng.normal(6.5 * k, .4),
                    epaisseur_mm=rng.normal(3.2 * k, .3), poids_mg=rng.normal(190 * k, 15)), "démo simulée")

def page_dashboard():
    st.header("🌿 Tableau de bord")
    n = {t: int(q(f"SELECT COUNT(*) n FROM {t}").n[0]) for t in ["arbres", *TRAITS]}
    c = st.columns(4)
    for col, (lab, k) in zip(c, [("Arbres", "arbres"), ("Gousses", "gousses"), ("Feuilles", "feuilles"), ("Graines", "graines")]):
        col.metric(lab, n[k])
    if n["arbres"] == 0:
        st.info("Base vide.")
        if st.button("Charger des données de démonstration (SIMULÉES)"):
            demo(); st.rerun()
        return
    t = st.selectbox("Organe", list(TRAITS), format_func=lambda x: TRAITS[x][0])
    d = load(t)
    if d.empty:
        st.info("Aucune mesure."); return
    cols = [c for c in d.select_dtypes("number").columns if c not in ("id", "arbre_id")]
    v = st.selectbox("Descripteur", cols)
    st.plotly_chart(px.histogram(d, x=v, color="cultivar", barmode="overlay", opacity=.7, nbins=30), width="stretch")

def page_arbres():
    st.header("🌳 Arbres / accessions")
    with st.form("arbre"):
        c = st.columns(3)
        code = c[0].text_input("Code unique*"); cv = c[1].selectbox("Cultivar", CULTIVARS); sx = c[2].selectbox("Sexe", SEXES)
        c = st.columns(4)
        lat = c[0].number_input("Latitude", value=35.0, format="%.5f"); lon = c[1].number_input("Longitude", value=2.0, format="%.5f")
        alt = c[2].number_input("Altitude (m)", 0.0, 3000.0, 300.0); age = c[3].number_input("Âge (ans)", 0, 1000, 20)
        notes = st.text_area("Notes")
        if st.form_submit_button("➕ Ajouter") and code:
            try:
                ex("INSERT INTO arbres(code,cultivar,sexe,latitude,longitude,altitude_m,age_ans,notes) VALUES(?,?,?,?,?,?,?,?)",
                   (code, cv, sx, lat, lon, alt, age, notes)); st.rerun()
            except sqlite3.IntegrityError:
                st.error("Ce code existe déjà.")
    d = q("SELECT * FROM arbres")
    st.dataframe(d, width="stretch", hide_index=True)
    if not d.empty:
        st.map(d.dropna(subset=["latitude", "longitude"]), latitude="latitude", longitude="longitude")
        sel = st.selectbox("Supprimer un arbre (et ses mesures)", arbre_opts())
        if st.button("🗑️ Supprimer"):
            ex("DELETE FROM arbres WHERE id=?", (arbre_opts()[sel],)); st.rerun()

def page_saisie():
    st.header("📝 Saisie des mesures (tableau)")
    opts = arbre_opts()
    if not opts:
        st.warning("Créez d'abord un arbre."); return
    aid = opts[st.selectbox("Arbre", opts)]
    tabs = st.tabs([TRAITS[t][0] for t in TRAITS])
    for tab, t in zip(tabs, TRAITS):
        with tab:
            cols = [c for c, _ in TRAITS[t][1]]
            cfg = {c: st.column_config.NumberColumn(l) for c, l in TRAITS[t][1]}
            ed = st.data_editor(pd.DataFrame(columns=cols, dtype=float), num_rows="dynamic",
                                column_config=cfg, key=f"ed_{t}", width="stretch")
            if st.button("💾 Enregistrer", key=f"sv_{t}"):
                ed = ed.apply(pd.to_numeric, errors="coerce").dropna(how="all")
                for _, r in ed.iterrows():
                    ins(t, aid, r.to_dict())
                st.success(f"{len(ed)} ligne(s) enregistrée(s)."); st.rerun()
            st.caption("Une ligne = un organe (1 gousse, 1 foliole, 1 graine). Mesurer ≥ 10 organes par arbre.")
            d = add_indices(t, q(f"SELECT * FROM {t} WHERE arbre_id=?", (aid,)))
            if not d.empty:
                st.dataframe(d.round(2), width="stretch", hide_index=True)

def page_photo():
    st.header("📷 Photo-mesure calibrée par pièce")
    if not CV2:
        st.error("OpenCV requis : pip install opencv-python-headless"); return
    st.caption("Photo à la verticale, fond clair uni, organes dans le même plan que la pièce, sans chevauchement. "
               "Mesure par contours (aucune estimation par ratios).")
    opts = arbre_opts()
    c = st.columns(4)
    coin = c[0].number_input("Diamètre pièce (mm) – à vérifier sur votre pièce", 10.0, 40.0, 25.0, 0.1)
    organe = c[1].selectbox("Organe", list(TRAITS), format_func=lambda x: TRAITS[x][0])
    min_mm2 = c[2].number_input("Aire min. objet (mm²)", 1.0, 5000.0, 20.0)
    aid = opts[c[3].selectbox("Arbre", opts)] if opts else None
    f = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
    if f and st.button("🔍 Mesurer", type="primary"):
        st.session_state["photo"] = measure_photo(f.read(), coin, min_mm2)
    if "photo" in st.session_state:
        objs, img, msg = st.session_state["photo"]
        st.image(img, caption=msg, width="stretch")
        if objs:
            df = pd.DataFrame([{k: round(v, 2) for k, v in o.items() if k != "k"} for o in objs], index=range(1, len(objs) + 1))
            st.dataframe(df, width="stretch")
            if aid and st.button("💾 Enregistrer tous les objets"):
                for o in objs:
                    v = dict(gousses=dict(longueur_mm=o["Lcurv"], largeur_mm=o["W"]),
                             feuilles=dict(long_foliole_mm=o["L"], larg_foliole_mm=o["W"]),
                             graines=dict(longueur_mm=o["L"], largeur_mm=o["W"]))[organe]
                    ins(organe, aid, v, "photo")
                st.success("Enregistré (épaisseur/poids à compléter dans « Saisie »).")

def page_analyses():
    st.header("📊 Analyses multivariées")
    w = wide()
    if len(w) < 3:
        st.info("Au moins 3 arbres avec mesures sont nécessaires."); return
    cols = [c for c in w.columns if c not in ("code", "cultivar")]
    t1, t2, t3, t4 = st.tabs(["Statistiques", "Corrélations", "ACP", "Classification"])
    with t1:
        sel = st.multiselect("Descripteurs", cols, default=cols[:4])
        if sel:
            st.dataframe(w.groupby("cultivar")[sel].agg(["mean", "std", "count"]).round(2), width="stretch")
            v = st.selectbox("Boxplot", sel)
            st.plotly_chart(px.box(w, x="cultivar", y=v, points="all", hover_name="code"), width="stretch")
    with t2:
        st.plotly_chart(px.imshow(w[cols].corr().round(2), text_auto=True, zmin=-1, zmax=1,
                                  color_continuous_scale="RdBu_r", aspect="auto"), width="stretch")
    X = w[cols].dropna(axis=1, how="all")
    X = X.loc[:, X.std() > 0].dropna()
    with t3:
        if X.shape[0] < 3 or X.shape[1] < 2:
            st.info("Données insuffisantes (valeurs manquantes ou variance nulle)."); return
        Z = (X - X.mean()) / X.std(ddof=0)
        U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
        var = S ** 2 / (S ** 2).sum() * 100
        sc = pd.DataFrame(U[:, :2] * S[:2], columns=["PC1", "PC2"], index=Z.index).join(w[["code", "cultivar"]])
        st.plotly_chart(px.scatter(sc, x="PC1", y="PC2", color="cultivar", hover_name="code",
                        labels={"PC1": f"PC1 ({var[0]:.1f}%)", "PC2": f"PC2 ({var[1]:.1f}%)"}), width="stretch")
        ld = pd.DataFrame(Vt[:2].T, index=Z.columns, columns=["PC1", "PC2"]).round(3)
        st.markdown("**Contributions des variables (loadings)**")
        st.dataframe(ld.sort_values("PC1", key=abs, ascending=False), width="stretch")
    with t4:
        if not SCIPY:
            st.error("pip install scipy"); return
        fig = ff.create_dendrogram(Z.values, labels=list(w.loc[Z.index, "code"]), orientation="left")
        fig.update_layout(height=max(400, 22 * len(Z)))
        st.plotly_chart(fig, width="stretch")
        st.caption("Distance euclidienne sur variables centrées-réduites (Ward par défaut de Plotly = complete linkage).")

def page_donnees():
    st.header("💾 Import / Export CSV")
    t = st.selectbox("Table", ["arbres", *TRAITS])
    d = q(f"SELECT * FROM {t}")
    st.download_button("⬇️ Exporter", d.to_csv(index=False).encode(), f"{t}.csv", "text/csv")
    f = st.file_uploader("Importer un CSV (colonnes = noms SQL de la table)", type="csv")
    if f is not None:
        df = pd.read_csv(f)
        st.dataframe(df.head(), width="stretch")
        if st.button("✅ Importer"):
            ok = [c for c in df.columns if c in d.columns and c != "id"]
            with db() as c:
                df[ok].to_sql(t, c, if_exists="append", index=False)
            st.success(f"{len(df)} lignes importées."); st.rerun()

# ───────────── i18n (navigation + pages ajoutées) ─────────────
TX = {
 "fr": dict(dashboard="🏠 Tableau de bord", arbres="🌳 Arbres", saisie="📝 Saisie", photo="📷 Photo-mesure",
            seg="🍃 Segmentation folioles", desc="📋 Descripteurs ordinaux", rend="📈 Rendement & phénologie",
            analyses="📊 Multivarié (ACP)", stats="🧪 ANOVA · Tukey · H²", molec="🧬 Marqueurs · Mantel",
            donnees="💾 Import/Export", lang="Langue", photo2="📐 Photo A4/pièce + couleur", arch="🌳 Architecture (bâton 1 m)"),
 "en": dict(dashboard="🏠 Dashboard", arbres="🌳 Trees", saisie="📝 Data entry", photo="📷 Photo measurement",
            seg="🍃 Leaflet segmentation", desc="📋 Ordinal descriptors", rend="📈 Yield & phenology",
            analyses="📊 Multivariate (PCA)", stats="🧪 ANOVA · Tukey · H²", molec="🧬 Markers · Mantel",
            donnees="💾 Import/Export", lang="Language", photo2="📐 A4/coin photo + colour", arch="🌳 Tree architecture (1 m stick)"),
 "ar": dict(dashboard="🏠 لوحة القيادة", arbres="🌳 الأشجار", saisie="📝 إدخال القياسات", photo="📷 القياس بالصورة",
            seg="🍃 تجزئة الوريقات", desc="📋 الواصفات الترتيبية", rend="📈 الإنتاج والفينولوجيا",
            analyses="📊 التحليل متعدد المتغيرات", stats="🧪 ANOVA · Tukey · H²", molec="🧬 الواسمات · Mantel",
            donnees="💾 استيراد/تصدير", lang="اللغة", photo2="📐 قياس A4/قطعة + اللون", arch="🌳 بنية الشجرة (عصا 1 م)"),
}
LANGS = {"fr": "Français", "en": "English", "ar": "العربية"}

def lang():
    return st.session_state.get("lang", "fr")

def tr(k):
    return TX[lang()].get(k, TX["fr"].get(k, k))

def sidebar_lang():
    st.sidebar.selectbox("🌐 " + tr("lang"), list(LANGS), format_func=LANGS.get, key="lang")
    if lang() == "ar":
        st.markdown("<style>.stApp,[data-testid=stSidebar]{direction:rtl;text-align:right}</style>",
                    unsafe_allow_html=True)

# ───────────── Constantes ─────────────
# Fiche de notation ordinale 1-9 : base de travail à aligner sur la liste officielle utilisée
DESC = ["Port de l'arbre", "Vigueur", "Densité du feuillage", "Forme de la foliole",
        "Couleur de la gousse (maturité)", "Courbure de la gousse", "Rugosité de la gousse",
        "Couleur de la graine", "Époque de maturité"]
STADES = ["Débourrement", "Début floraison", "Pleine floraison", "Nouaison",
          "Grossissement des gousses", "Maturité des gousses"]


def init_db_extras(h):
    h.ex("""CREATE TABLE IF NOT EXISTS ordinales(id INTEGER PRIMARY KEY AUTOINCREMENT,
         arbre_id INTEGER REFERENCES arbres(id) ON DELETE CASCADE, annee INTEGER, descripteur TEXT, score INTEGER)""")
    h.ex("""CREATE TABLE IF NOT EXISTS phenologie(id INTEGER PRIMARY KEY AUTOINCREMENT,
         arbre_id INTEGER REFERENCES arbres(id) ON DELETE CASCADE, annee INTEGER, stade TEXT, date_obs TEXT)""")
    h.ex("""CREATE TABLE IF NOT EXISTS rendement(id INTEGER PRIMARY KEY AUTOINCREMENT,
         arbre_id INTEGER REFERENCES arbres(id) ON DELETE CASCADE, annee INTEGER, kg REAL, surface_cime_m2 REAL,
         UNIQUE(arbre_id, annee))""")
    h.ex("""CREATE TABLE IF NOT EXISTS marqueurs(id INTEGER PRIMARY KEY AUTOINCREMENT,
         arbre_id INTEGER REFERENCES arbres(id) ON DELETE CASCADE, locus TEXT, a1 TEXT, a2 TEXT,
         UNIQUE(arbre_id, locus))""")


# ───────────── Statistiques ─────────────
def var_comp(y, g):
    """ANOVA à 1 facteur aléatoire, effectifs déséquilibrés → F, p, eta², composantes de variance, ICC, H² (moyenne)."""
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    grp = d.groupby("g").y
    n = grp.size(); k = len(n); N = len(d)
    if k < 2 or N - k < 1:
        return None
    ssb = (n * (grp.mean() - d.y.mean()) ** 2).sum()
    ssw = ((d.y - grp.transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    s2g = max((msb - msw) / n0, 0)
    F = msb / msw if msw > 0 else np.nan
    return dict(F=F, p=stats.f.sf(F, k - 1, N - k) if SCIPY and msw > 0 else np.nan,
                eta2=ssb / (ssb + ssw), s2_entre=s2g, s2_intra=msw,
                ICC=s2g / (s2g + msw) if s2g + msw > 0 else np.nan,
                H2_moyenne=max((msb - msw) / msb, 0) if msb > 0 else np.nan)


def mantel(A, B, n=999, seed=0):
    iu = np.triu_indices_from(A, 1)
    a = A[iu]; r = np.corrcoef(a, B[iu])[0, 1]
    rng = np.random.default_rng(seed); c = 0
    for _ in range(n):
        p = rng.permutation(len(B))
        c += abs(np.corrcoef(a, B[np.ix_(p, p)][iu])[0, 1]) >= abs(r)
    return r, (c + 1) / (n + 1)


def dps(m):
    """Distance = 1 − proportion d'allèles partagés (diploïde ; valable SSR et SNP)."""
    G = {c: {} for c in m.code.unique()}
    for r in m.itertuples():
        if pd.notna(r.a1) and pd.notna(r.a2) and str(r.a1) != "":
            G[r.code][r.locus] = [str(r.a1), str(r.a2)]
    codes = sorted(G); n = len(codes); D = np.full((n, n), np.nan)
    for i, j in itertools.combinations_with_replacement(range(n), 2):
        sh = [sum(min(G[codes[i]][l].count(x), G[codes[j]][l].count(x)) for x in set(G[codes[i]][l] + G[codes[j]][l])) / 2
              for l in set(G[codes[i]]) & set(G[codes[j]])]
        if sh:
            D[i, j] = D[j, i] = 1 - np.mean(sh)
    return codes, D


# ───────────── Segmentation de folioles ─────────────
def segment_leaflets(data, coin_mm, hmin, hmax, smin, vmin, min_mm2, split):
    """Pièce étalon (Hough) → masque HSV vert → raffinement GrabCut → séparation par watershed → mesures."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    s = 1400 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s)
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=25,
                         minRadius=15, maxRadius=int(min(img.shape[:2]) / 4))
    if c is None:
        return [], cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Pièce étalon non détectée."
    x, y, r = c[0][0]; pxmm = 2 * r / coin_mm
    m = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (hmin, smin, vmin), (hmax, 255, 255))
    cv2.circle(m, (int(x), int(y)), int(r * 1.2), 0, -1)
    k5 = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k5), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(m, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(img, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
        m = np.where((gc == 1) | (gc == 3), 255, 0).astype(np.uint8)
    except cv2.error:
        pass
    if split:
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        _, fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
        fg = fg.astype(np.uint8)
        n, mk = cv2.connectedComponents(fg)
        mk = mk + 1; mk[cv2.subtract(m, fg) > 0] = 0
        mk = cv2.watershed(img, mk.astype(np.int32))
        masks = [(mk == i).astype(np.uint8) * 255 for i in range(2, n + 1)]
    else:
        n, mk = cv2.connectedComponents(m)
        masks = [(mk == i).astype(np.uint8) * 255 for i in range(1, n)]
    objs = []
    for mm in masks:
        cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        k = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(k) / pxmm ** 2
        if area < min_mm2:
            continue
        (_, _), (w, hh), _ = cv2.minAreaRect(k)
        hull = cv2.contourArea(cv2.convexHull(k)) / pxmm ** 2
        objs.append(dict(L=max(w, hh) / pxmm, W=min(w, hh) / pxmm, aire_mm2=area,
                         solidite=area / hull if hull else np.nan, k=k))
    cv2.circle(img, (int(x), int(y)), int(r), (0, 200, 0), 3)
    for i, o in enumerate(objs, 1):
        cv2.drawContours(img, [o["k"]], -1, (0, 0, 255), 2)
        bx, by, _, _ = cv2.boundingRect(o["k"])
        cv2.putText(img, str(i), (bx, max(by - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 2)
    return objs, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"{pxmm:.2f} px/mm · {len(objs)} foliole(s)"


# ───────────── Calibration A4 / pièce + colorimétrie ─────────────
S = 6  # px/mm après redressement de l'A4

def calibrate(img, mode, coin_mm):
    """A4 : redressement de perspective (échelle exacte, 6 px/mm). Pièce : échelle par cercle de Hough."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "A4":
        _, th = cv2.threshold(cv2.GaussianBlur(g, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for k in sorted(cs, key=cv2.contourArea, reverse=True)[:3]:
            a = cv2.approxPolyDP(k, 0.02 * cv2.arcLength(k, True), True)
            if len(a) == 4 and cv2.contourArea(k) > 0.15 * g.size:
                p = a.reshape(4, 2).astype(np.float32); s, d = p.sum(1), np.diff(p, axis=1).ravel()
                P = np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)  # tl,tr,br,bl
                W, H = (210, 297) if np.linalg.norm(P[3] - P[0]) >= np.linalg.norm(P[1] - P[0]) else (297, 210)
                M = cv2.getPerspectiveTransform(P, np.array([[0, 0], [W * S, 0], [W * S, H * S], [0, H * S]], np.float32))
                m = 8 * S  # marge de 8 mm (bords/ombres du papier)
                return cv2.warpPerspective(img, M, (W * S, H * S))[m:-m, m:-m], S, "A4 détectée, perspective corrigée", None
        return None, 0, "A4 non détectée : 4 coins visibles, papier plus clair que le fond.", None
    c = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=25,
                         minRadius=15, maxRadius=int(min(g.shape) / 4))
    if c is None:
        return None, 0, "Pièce non détectée.", None
    x, y, r = c[0][0]
    return img, 2 * r / coin_mm, f"Pièce détectée : {2 * r / coin_mm:.2f} px/mm (pas de correction de perspective ni de couleur)", (x, y, r)


def analyse(img, px, wb, min_mm2, coin=None):
    """Mesures (L, l, longueur curviligne, aire, solidité) + couleur CIELAB (blanc du papier = référence)."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if coin:
        cv2.circle(m, (int(coin[0]), int(coin[1])), int(coin[2] * 1.2), 0, -1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if wb and (m == 0).any():
        img = np.clip(img * (240 / np.median(img[m == 0], axis=0)), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    for k in cs:
        area = cv2.contourArea(k) / px ** 2
        if area < min_mm2:
            continue
        (_, _), (w, hh), _ = cv2.minAreaRect(k); L, W = max(w, hh) / px, min(w, hh) / px
        mk = np.zeros(m.shape, np.uint8); cv2.drawContours(mk, [k], -1, 255, -1)
        mk = cv2.erode(mk, np.ones((5, 5), np.uint8), iterations=2)
        if not mk.any():
            continue
        l, a, b = lab[mk > 0].mean(0)
        objs.append(dict(L=L, W=W, Lcurv=max(cv2.arcLength(k, True) / px / 2 - W, L), aire_mm2=area,
                         solidite=area / (cv2.contourArea(cv2.convexHull(k)) / px ** 2),
                         L_star=l * 100 / 255, a_star=a - 128, b_star=b - 128, k=k))
    out = img.copy()
    for i, o in enumerate(objs, 1):
        cv2.drawContours(out, [o["k"]], -1, (0, 0, 255), 2)
        bx, by, _, _ = cv2.boundingRect(o["k"])
        cv2.putText(out, str(i), (bx, max(by - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 0, 0), 2)
    return objs, cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ───────────── Pages ─────────────
def build_pages(h):
    def pick():
        o = h.arbre_opts()
        if not o:
            st.warning("Créez d'abord un arbre."); return None, None
        return o, o[st.selectbox("Arbre", o)]

    def p_desc():
        st.header(tr("desc"))
        st.caption("Notation ordinale 1–9 (format UPOV/Bioversity). Les intitulés sont une base de travail : "
                   "alignez-les sur la liste officielle de descripteurs « caroubier » que vous suivez.")
        o, aid = pick()
        if not o: return
        an = st.number_input("Année", 1990, 2100, datetime.date.today().year)
        cur = h.q("SELECT descripteur,score FROM ordinales WHERE arbre_id=? AND annee=?", (aid, an)).set_index("descripteur").score.to_dict()
        ed = st.data_editor(pd.DataFrame({"descripteur": DESC, "score": [float(cur.get(d, np.nan)) for d in DESC]}),
                            column_config={"descripteur": st.column_config.TextColumn(disabled=True),
                                           "score": st.column_config.NumberColumn(min_value=1, max_value=9, step=1)},
                            hide_index=True, width="stretch")
        if st.button("💾 Enregistrer"):
            h.ex("DELETE FROM ordinales WHERE arbre_id=? AND annee=?", (aid, an))
            for r in ed.dropna().itertuples():
                h.ex("INSERT INTO ordinales(arbre_id,annee,descripteur,score) VALUES(?,?,?,?)", (aid, an, r.descripteur, int(r.score)))
            st.success("Enregistré.")
        d = h.q("SELECT a.cultivar,o.descripteur,AVG(o.score) m FROM ordinales o JOIN arbres a ON a.id=o.arbre_id GROUP BY 1,2")
        if not d.empty:
            st.plotly_chart(px.imshow(d.pivot(index="descripteur", columns="cultivar", values="m").round(1), text_auto=True,
                                      color_continuous_scale="YlGn", aspect="auto"), width="stretch")

    def p_rend():
        st.header(tr("rend"))
        t1, t2 = st.tabs(["Rendement", "Phénologie"])
        with t1:
            o, aid = pick()
            if not o: return
            ed = st.data_editor(pd.DataFrame({"annee": pd.Series(dtype="float"), "kg": pd.Series(dtype="float"),
                                              "surface_cime_m2": pd.Series(dtype="float")}), num_rows="dynamic", width="stretch")
            if st.button("💾 Enregistrer", key="sv_rend"):
                for r in ed.dropna(subset=["annee", "kg"]).itertuples():
                    h.ex("INSERT OR REPLACE INTO rendement(arbre_id,annee,kg,surface_cime_m2) VALUES(?,?,?,?)",
                         (aid, int(r.annee), r.kg, None if pd.isna(r.surface_cime_m2) else r.surface_cime_m2))
                st.rerun()
            d = h.q("SELECT r.*,a.code,a.cultivar FROM rendement r JOIN arbres a ON a.id=r.arbre_id ORDER BY a.code,annee")
            if not d.empty:
                d["kg_par_m2_cime"] = d.kg / d.surface_cime_m2
                st.dataframe(d.drop(columns=["id", "arbre_id"]).round(2), width="stretch", hide_index=True)
                st.plotly_chart(px.line(d, x="annee", y="kg", color="code", markers=True), width="stretch")
                # indice d'alternance : moyenne de |y(t+1)-y(t)| / (y(t+1)+y(t)) ; 0 = régulier, 1 = alternance maximale
                alt = {c: np.nanmean(np.abs(np.diff(g.kg.values)) / (g.kg.values[1:] + g.kg.values[:-1]))
                       for c, g in d.groupby("code") if len(g) > 1}
                if alt:
                    st.markdown("**Indice d'alternance de production (0 = régulier → 1 = fortement alternant)**")
                    st.dataframe(pd.Series(alt, name="indice").round(3), width="stretch")
        with t2:
            o, aid = pick() if False else (h.arbre_opts(), None)
            if not o: return
            with st.form("pheno"):
                c = st.columns(4)
                a = o[c[0].selectbox("Arbre", o, key="ph_a")]; an = c[1].number_input("Année", 1990, 2100, datetime.date.today().year, key="ph_y")
                stg = c[2].selectbox("Stade", STADES); dt = c[3].date_input("Date d'observation")
                if st.form_submit_button("➕ Ajouter"):
                    h.ex("INSERT INTO phenologie(arbre_id,annee,stade,date_obs) VALUES(?,?,?,?)", (a, int(an), stg, str(dt))); st.rerun()
            d = h.q("SELECT p.*,a.code,a.cultivar FROM phenologie p JOIN arbres a ON a.id=p.arbre_id")
            if not d.empty:
                d["jour_annee"] = pd.to_datetime(d.date_obs).dt.dayofyear
                st.plotly_chart(px.box(d, x="stade", y="jour_annee", color="cultivar", points="all", hover_name="code",
                                       category_orders={"stade": STADES}), width="stretch")

    def p_stats():
        st.header(tr("stats"))
        if not SCIPY:
            st.error("pip install scipy"); return
        t1, t2, t3 = st.tabs(["Répétabilité (ICC)", "ANOVA + Tukey", "Héritabilité"])
        with t1:
            st.caption("ICC = part de variance entre arbres sur l'ensemble des organes mesurés (répétitions : gousses, folioles, graines). "
                       "Un ICC élevé et un CV intra bas = protocole de mesure fiable.")
            rows = []
            for t in h.TRAITS:
                d = h.load(t)
                for c in [c for c in d.select_dtypes("number").columns if c not in ("id", "arbre_id")]:
                    v = var_comp(d[c], d.arbre_id)
                    cv = (d.groupby("arbre_id")[c].std() / d.groupby("arbre_id")[c].mean().abs() * 100).mean()
                    if v: rows.append(dict(organe=t, descripteur=c, ICC=v["ICC"], CV_intra_pct=cv, p=v["p"]))
            st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True) if rows else st.info("Pas assez de répétitions (≥ 2 arbres, ≥ 2 organes).")
        w = h.wide()
        cols = [c for c in w.columns if c not in ("code", "cultivar")] if not w.empty else []
        with t2:
            if not cols: st.info("Données insuffisantes."); return
            rows = []
            for c in cols:
                v = var_comp(w[c], w.cultivar)
                if v: rows.append(dict(descripteur=c, F=v["F"], p=v["p"], eta2=v["eta2"]))
            if rows: st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
            st.caption("Unité statistique = moyenne par arbre (évite la pseudo-réplication des gousses).")
            tr_ = st.selectbox("Tukey HSD sur", cols)
            grp = {k: s.dropna().values for k, s in w.groupby("cultivar")[tr_]}
            grp = {k: v for k, v in grp.items() if len(v) >= 2}
            if len(grp) >= 2:
                res = stats.tukey_hsd(*grp.values()); ks = list(grp)
                pairs = [dict(A=ks[i], B=ks[j], diff_moy=grp[ks[i]].mean() - grp[ks[j]].mean(), p_ajust=res.pvalue[i, j],
                              signif="✱" if res.pvalue[i, j] < .05 else "") for i, j in itertools.combinations(range(len(ks)), 2)]
                st.dataframe(pd.DataFrame(pairs).round(4), width="stretch", hide_index=True)
            else:
                st.info("≥ 2 cultivars avec ≥ 2 arbres chacun.")
        with t3:
            if not cols: return
            st.warning("H² estimée = (MSG−MSE)/MSG entre cultivars, arbres = répétitions. Sans réplication clonale ni verger "
                       "comparatif, génotype et environnement sont confondus : valeur indicative (borne haute), pas une H² stricte.")
            rows = []
            for c in cols:
                v = var_comp(w[c], w.cultivar)
                if v: rows.append(dict(descripteur=c, var_cultivar=v["s2_entre"], var_residuelle=v["s2_intra"], H2=v["H2_moyenne"]))
            if rows: st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)

    def p_molec():
        st.header(tr("molec"))
        st.caption("Format long : un locus par ligne. SSR : tailles d'allèles (ex. 152 / 158). SNP : bases (A / G). Homozygote : même valeur deux fois.")
        ids = {r.code: int(r.id) for r in h.q("SELECT id,code FROM arbres").itertuples()}
        f = st.file_uploader("CSV (code, locus, a1, a2)", type="csv")
        base = pd.read_csv(f, dtype=str) if f else pd.DataFrame({"code": pd.Series(dtype=str), "locus": pd.Series(dtype=str),
                                                               "a1": pd.Series(dtype=str), "a2": pd.Series(dtype=str)})
        ed = st.data_editor(base, num_rows="dynamic", width="stretch")
        if st.button("💾 Enregistrer les génotypes"):
            bad = set(ed.code.dropna()) - set(ids)
            if bad: st.error(f"Codes d'arbres inconnus : {sorted(bad)}")
            else:
                for r in ed.dropna().itertuples():
                    h.ex("INSERT OR REPLACE INTO marqueurs(arbre_id,locus,a1,a2) VALUES(?,?,?,?)", (ids[r.code], r.locus, str(r.a1), str(r.a2)))
                st.success("Enregistré.")
        m = h.q("SELECT a.code,k.locus,k.a1,k.a2 FROM marqueurs k JOIN arbres a ON a.id=k.arbre_id")
        if m.empty: return
        div = []
        for l, g in m.groupby("locus"):
            al = pd.concat([g.a1, g.a2]).astype(str); p = al.value_counts(normalize=True).values
            pic = 1 - (p ** 2).sum() - sum(2 * p[i] ** 2 * p[j] ** 2 for i in range(len(p)) for j in range(i + 1, len(p)))
            div.append(dict(locus=l, n_genotypes=len(g), Na=len(p), Ho=(g.a1 != g.a2).mean(), He=1 - (p ** 2).sum(), PIC=pic))
        st.markdown("**Diversité par locus**"); st.dataframe(pd.DataFrame(div).round(3), width="stretch", hide_index=True)
        codes, D = dps(m)
        if len(codes) < 4: st.info("≥ 4 arbres génotypés pour Mantel."); return
        st.plotly_chart(px.imshow(pd.DataFrame(D, index=codes, columns=codes).round(3), color_continuous_scale="Viridis",
                                  title="Distance génétique (1 − allèles partagés)"), width="stretch")
        w = h.wide()
        if w.empty: return
        w = w.set_index("code"); common = [c for c in codes if c in w.index]
        X = w.loc[common].drop(columns="cultivar").dropna(axis=1)
        X = X.loc[:, X.std() > 0]
        if len(common) < 4 or X.shape[1] < 2: st.info("Morphométrie insuffisante sur les arbres génotypés."); return
        Z = ((X - X.mean()) / X.std(ddof=0)).values
        Dm = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        idx = [codes.index(c) for c in common]; Dg = D[np.ix_(idx, idx)].copy()
        Dg[np.isnan(Dg)] = np.nanmean(Dg)
        r, p = mantel(Dg, Dm)
        c1, c2 = st.columns(2); c1.metric("Mantel r", f"{r:.3f}"); c2.metric("p (999 permutations, bilatéral)", f"{p:.3f}")
        iu = np.triu_indices(len(common), 1)
        st.plotly_chart(px.scatter(x=Dg[iu], y=Dm[iu], trendline="ols" if SCIPY and False else None,
                                   labels={"x": "Distance génétique", "y": "Distance morphologique (euclid. standardisée)"}),
                        width="stretch")

    def p_seg():
        st.header(tr("seg"))
        if not CV2: st.error("pip install opencv-python-headless"); return
        st.caption("Fond complexe : masque HSV vert + GrabCut + watershed (folioles qui se touchent). "
                   "Ajustez la plage de teinte si le masque est incomplet. Pièce étalon dans le plan des feuilles.")
        o = h.arbre_opts()
        c = st.columns(4)
        coin = c[0].number_input("Diamètre pièce (mm)", 10.0, 40.0, 25.0, .1, key="sg_c")
        hr = c[1].slider("Teinte (H)", 0, 179, (25, 95)); sm = c[2].slider("Saturation min", 0, 255, 40); vm = c[3].slider("Valeur min", 0, 255, 30)
        c = st.columns(3)
        mn = c[0].number_input("Aire min. (mm²)", 5.0, 5000.0, 100.0); sp = c[1].checkbox("Séparer les folioles jointives", True)
        aid = o[c[2].selectbox("Arbre", o)] if o else None
        f = st.file_uploader("Photo de feuilles", type=["jpg", "jpeg", "png"], key="sg_f")
        if f and st.button("🍃 Segmenter et mesurer", type="primary"):
            with st.spinner("Segmentation…"):
                st.session_state["seg"] = segment_leaflets(f.read(), coin, hr[0], hr[1], sm, vm, mn, sp)
        if "seg" in st.session_state:
            objs, img, msg = st.session_state["seg"]
            st.image(img, caption=msg, width="stretch")
            if objs:
                st.dataframe(pd.DataFrame([{k: round(v, 2) for k, v in x.items() if k != "k"} for x in objs],
                                          index=range(1, len(objs) + 1)), width="stretch")
                st.caption("Solidité basse (< 0,85) = objet probablement fusionné ou mal segmenté : vérifier visuellement.")
                if aid and st.button("💾 Enregistrer dans « feuilles »"):
                    for x in objs:
                        h.ins("feuilles", aid, dict(long_foliole_mm=x["L"], larg_foliole_mm=x["W"]), "segmentation")
                    st.success("Enregistré.")

    def p_photo2():
        st.header(tr("photo2"))
        if not CV2: st.error("pip install opencv-python-headless"); return
        st.caption("Organes à plat, sans chevauchement, sur une feuille A4 blanche, lumière diffuse, photo de dessus avec les 4 coins "
                   "visibles : la perspective est corrigée et le blanc du papier sert de référence couleur. Graines : une pièce (mode « Pièce ») est possible.")
        c = st.columns(4)
        org = c[0].selectbox("Organe", ["gousses", "graines", "feuilles"], format_func=lambda x: h.TRAITS[x][0])
        mode = c[1].radio("Étalon", ["A4", "Pièce"], horizontal=True)
        coin = c[2].number_input("Diamètre pièce (mm)", 10.0, 40.0, 25.0, .1, key="p2c")
        mn = c[3].number_input("Aire min. (mm²)", 1.0, 5000.0, {"gousses": 200.0, "graines": 8.0, "feuilles": 50.0}[org])
        o, aid = pick()
        f = st.file_uploader("Photo", type=["jpg", "jpeg", "png"], key="p2f")
        if f and st.button("📐 Mesurer", type="primary"):
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
            s = 3000 / max(img.shape[:2])
            if s < 1: img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            im, px, msg, cn = calibrate(img, mode, coin)
            st.session_state["p2"] = (*analyse(im, px, mode == "A4", mn, cn), msg) if im is not None else ([], None, msg)
        if "p2" in st.session_state:
            objs, ov, msg = st.session_state["p2"]
            st.info(msg)
            if ov is not None: st.image(ov, width="stretch")
            if objs:
                df = pd.DataFrame([{k: round(float(v), 2) for k, v in x.items() if k != "k"} for x in objs], index=range(1, len(objs) + 1))
                df["chroma_C"] = np.hypot(df.a_star, df.b_star).round(1)
                df["teinte_h"] = (np.degrees(np.arctan2(df.b_star, df.a_star)) % 360).round(0)
                st.dataframe(df, width="stretch")
                if aid and st.button("💾 Enregistrer", key="p2s"):
                    for x in objs:
                        v = dict(gousses=dict(longueur_mm=x["Lcurv"], largeur_mm=x["W"]), graines=dict(longueur_mm=x["L"], largeur_mm=x["W"]),
                                 feuilles=dict(long_foliole_mm=x["L"], larg_foliole_mm=x["W"]))[org]
                        v.update(L_star=x["L_star"], a_star=x["a_star"], b_star=x["b_star"])
                        h.ins(org, aid, v, "photo " + ("A4" if mode == "A4" else "pièce"))
                    st.success("Enregistré (épaisseur, poids : à saisir dans « Saisie »).")

    def p_arch():
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates as sic
        except ImportError:
            st.error("pip install streamlit-image-coordinates"); return
        from PIL import Image, ImageDraw
        MES = {"hauteur_m": "Hauteur totale (sol → sommet)", "haut_tronc_m": "Hauteur du tronc (sol → 1ʳᵉ branche)",
               "diam_cime_1_m": "Largeur de cime (gauche ↔ droite)"}
        st.header(tr("arch"))
        st.caption("Bâton de 1 m dressé verticalement contre le tronc (même plan), photo à 10–20 m, appareil à mi-hauteur, bien d'aplomb. "
                   "Cliquez 2 points par mesure. Précision limitée par la perspective : indicatif pour les grands arbres (télémètre/clinomètre sinon).")
        o, aid = pick()
        if not o: return
        f = st.file_uploader("Photo de l'arbre + bâton", type=["jpg", "jpeg", "png"], key="ar_f")
        if not f: return
        c = st.columns([1, 3, 1])
        Ls = c[0].number_input("Bâton (m)", .1, 5.0, 1.0, .05)
        tgt = c[1].radio("Points à cliquer", ["stick", *MES], horizontal=True,
                         format_func=lambda k: "Bâton (2 extrémités)" if k == "stick" else MES[k])
        if c[2].button("↺ Reset"): st.session_state.pop("pts", None); st.rerun()
        pts = st.session_state.setdefault("pts", {})
        img = Image.open(f).convert("RGB"); W0 = img.width; dr = ImageDraw.Draw(img)
        for k, v in pts.items():
            for p in v: dr.ellipse([p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8], outline="red", width=4)
            if len(v) == 2: dr.line(v[0] + v[1], fill="yellow" if k == "stick" else "cyan", width=4)
        cl = sic(img, width=700, key="sic")
        if cl and cl != st.session_state.get("last"):
            st.session_state["last"] = cl; sc = W0 / cl["width"]; p = (cl["x"] * sc, cl["y"] * sc)
            v = pts.get(tgt, []); pts[tgt] = [p] if len(v) >= 2 else v + [p]; st.rerun()
        dist = lambda v: float(np.hypot(v[0][0] - v[1][0], v[0][1] - v[1][1]))
        if len(pts.get("stick", [])) < 2: st.info("Cliquez d'abord les 2 extrémités du bâton."); return
        pxm = dist(pts["stick"]) / Ls
        vals = {k: round(dist(pts[k]) / pxm, 2) for k in MES if len(pts.get(k, [])) == 2}
        c = st.columns(3)
        d2 = c[0].number_input("Diamètre cime 2 (m, perpendiculaire)", 0.0, 30.0, 0.0, .1)
        circ = c[1].number_input("Circonférence tronc à 1,3 m (cm, au mètre-ruban)", 0.0, 1000.0, 0.0, 1.0)
        vals.update(diam_cime_2_m=d2 or None, circ_tronc_cm=circ or None)
        st.json(vals)
        if st.button("💾 Enregistrer"): h.ins("architecture", aid, vals, "photo+bâton"); st.success("Enregistré.")

    return dict(desc=p_desc, rend=p_rend, stats=p_stats, molec=p_molec, seg=p_seg, photo2=p_photo2, arch=p_arch)


H = SimpleNamespace(q=q, ex=ex, ins=ins, load=load, wide=wide, arbre_opts=arbre_opts, TRAITS=TRAITS)
PAGES = {"dashboard": page_dashboard, "arbres": page_arbres, "saisie": page_saisie, "photo": page_photo,
         "analyses": page_analyses, "donnees": page_donnees, **build_pages(H)}
ORDER = ["dashboard", "arbres", "saisie", "photo2", "photo", "seg", "arch", "desc", "rend", "analyses", "stats", "molec", "donnees"]

def main():
    init_db(); migrate(); init_db_extras(H)
    st.sidebar.title("🌿 CaroubTrack")
    st.sidebar.caption("Ceratonia siliqua L.")
    sidebar_lang()
    PAGES[st.sidebar.radio("Navigation", ORDER, format_func=tr)]()

if __name__ == "__main__":
    main()
