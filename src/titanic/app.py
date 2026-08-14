import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
import shap
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

st.title("🚢 Titanic Survival Predictor")
st.write("Simulez un passager du Titanic et estimez sa probabilité de survie.")

st.info(
    "Le service peut mettre quelques secondes à démarrer "
    "après une période d'inactivité."
)

st.sidebar.header("Profil du passager")

pclass = st.sidebar.selectbox("Classe", options=[1, 2, 3], index=2)
title = st.sidebar.selectbox("Titre", options=["Mr", "Mrs", "Miss", "Master"])
sex = st.sidebar.selectbox("Sexe", options=["male", "female"])
age = st.sidebar.slider("Âge", min_value=0, max_value=100, value=28)

travel_group_size = st.sidebar.slider(
    "Nombre de personnes voyageant ensemble",
    min_value=1,
    max_value=20,
    value=1,
)

has_cabin = st.sidebar.checkbox("Cabine connue", value=False)

payload = {
    "pclass": pclass,
    "title": title,
    "age": age,
    "sex": sex,
    "travel_group_size": travel_group_size,
    "has_cabin": has_cabin,
}

st.subheader("Paramètres envoyés à l'API")
st.json(payload)

if st.button("Prédire la survie"):
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=10,
    )

    if response.status_code != 200:
        st.error("Erreur API lors de la prédiction.")
        st.text(response.text)
        st.stop()

    result = response.json()

    probability = result["survival_probability"]
    label = result["label"]

    st.subheader("Résultat")

    st.metric(
        label="Probabilité de survie",
        value=f"{probability:.1%}",
    )

    if result["prediction"] == 1:
        st.success(label)
    else:
        st.error(label)

    st.progress(probability)

    explain_response = requests.post(
        f"{API_URL}/explain",
        json=payload,
        timeout=20,
    )

    if explain_response.status_code != 200:
        st.warning("Impossible de récupérer l'explication SHAP.")
        st.text(explain_response.text)
        st.stop()

    explanation = explain_response.json()
    factors = explanation["top_factors"]

    st.subheader("Explication de la prédiction")

    factors_df = pd.DataFrame(factors)
    factors_df["abs_contribution"] = factors_df["contribution"].abs()

    factors_df = factors_df.sort_values(
        "abs_contribution", ascending=False
    ).reset_index(drop=True)

    st.dataframe(
        factors_df[["feature", "contribution", "direction"]],
        width="stretch",
    )

    feature_values = {
        "Pclass": pclass,
        "Title": title,
        "HasNickname": 0,
        "AgeETR": age,
        "IsChild": int(age < 7),
        "FarePerPerson_log1p": None,
        "FamilySurvivalRate": None,
        "TicketSurvivalRate": None,
        "HasCabin": int(has_cabin),
        "IsAlone": int(travel_group_size == 1),
    }

    shap_values = factors_df["contribution"].to_numpy()
    feature_names = factors_df["feature"].tolist()
    data_values = [feature_values.get(feature) for feature in feature_names]

    base_value = explanation["survival_probability"] - factors_df["contribution"].sum()

    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=data_values,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap_explanation,
        max_display=len(feature_names),
        show=False,
    )

    st.pyplot(plt.gcf())
    plt.close()
