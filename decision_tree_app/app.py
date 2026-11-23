import numpy as np
import pandas as pd
import streamlit as st

from decision_tree import ArbreDecision, regles, profondeur, nb_feuilles
from decision_tree.tree_visualisation import visualize_tree
from data.dataset import X as DEMO_X, y as DEMO_Y, noms as DEMO_NOMS

MAX_ROWS = 3000  

st.set_page_config(page_title="Decision Tree Explorer", page_icon="🌳", layout="wide")

st.title("🌳 Decision Tree Explorer")
st.write("Importe un CSV (ou utilise le jeu de démonstration) et visualise ton arbre de décision.")


def show_plot(fig):
    """Compatible avec les anciennes et nouvelles versions de Streamlit."""
    try:
        st.plotly_chart(fig, width="content")
    except TypeError:
        st.plotly_chart(fig, use_container_width=False)


def show_df(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------- données
source = st.radio("Source des données", ["Jeu de démonstration", "Importer un CSV"],
                  horizontal=True)

df = None
if source == "Jeu de démonstration":
    df = pd.DataFrame(DEMO_X, columns=DEMO_NOMS)
    df["défaut"] = DEMO_Y
else:
    up = st.file_uploader("Dataset (CSV)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Lecture du CSV impossible : {e}")

if df is None:
    st.info("Charge un fichier CSV pour commencer.")
    st.stop()

st.subheader("Aperçu")
show_df(df.head(10))
st.caption(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")

# ---------------------------------------------------------- configuration
target = st.selectbox("Variable cible", df.columns,
                      index=len(df.columns) - 1)

numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
ignored = [c for c in df.columns if c != target and c not in numeric_cols]
if ignored:
    st.warning(f"Colonnes non numériques ignorées : {', '.join(map(str, ignored))}")

features = st.multiselect("Variables explicatives", numeric_cols, default=numeric_cols)

st.sidebar.header("Configuration de l'arbre")
criterion = st.sidebar.selectbox(
    "Critère", ["gini", "entropie", "mse"],
    help="gini / entropie : classification. mse : régression (cible numérique).")
max_depth = st.sidebar.slider("Profondeur maximale", 1, 10, 3)
min_samples_split = st.sidebar.number_input("min_samples_split", min_value=2, value=2)
min_samples_leaf = st.sidebar.number_input("min_samples_leaf", min_value=1, value=1)
drop_na = st.sidebar.checkbox("Supprimer les lignes avec valeurs manquantes", value=True)

# --------------------------------------------------------------- entraînement
if st.button("🌳 Générer l'arbre", type="primary"):

    if not features:
        st.error("Sélectionne au moins une variable explicative.")
        st.stop()

    data = df[features + [target]]
    if data.isnull().any().any():
        if drop_na:
            data = data.dropna()
            st.info(f"{len(df) - len(data)} ligne(s) avec valeurs manquantes supprimée(s).")
        else:
            st.error("Il reste des valeurs manquantes : traite-les ou coche la suppression.")
            st.stop()

    if len(data) == 0:
        st.error("Aucune ligne exploitable.")
        st.stop()

    if len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=0)
        st.info(f"Échantillon aléatoire de {MAX_ROWS} lignes utilisé (pour la vitesse).")

    if criterion == "mse" and not pd.api.types.is_numeric_dtype(data[target]):
        st.error("Le critère 'mse' (régression) nécessite une cible numérique.")
        st.stop()

    try:
        Xm = data[features].to_numpy(dtype=float)
        ym = data[target].to_numpy()
        if ym.dtype == object:
            ym = ym.astype(str)

        model = ArbreDecision(critere=criterion, max_depth=max_depth,
                              min_samples_split=int(min_samples_split),
                              min_samples_leaf=int(min_samples_leaf))
        model.fit(Xm, ym)
        score = model.score(Xm, ym)
    except Exception as e:
        st.error(f"Impossible de générer l'arbre : {e}")
        st.stop()

    st.success("Arbre généré !")

    c1, c2, c3 = st.columns(3)
    c1.metric("Profondeur", profondeur(model.racine))
    c2.metric("Feuilles", nb_feuilles(model.racine))
    c3.metric("R² (entraînement)" if model.regression else "Exactitude (entraînement)",
              f"{score:.3f}")

    st.subheader("Ton arbre de décision")
    show_plot(visualize_tree(model.racine, features, criterion=criterion))

    with st.expander("Règles de décision"):
        st.code("\n".join(regles(model.racine, features)), language="text")
