import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import shap
import streamlit as st

API_URL = st.secrets.get(
    "API_URL",
    os.getenv("API_URL", "http://127.0.0.1:8000"),
)


@st.cache_resource(ttl=300)
def wake_up_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=90,
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

st.title("🚢 Titanic Survival Predictor")
st.write("Simulez un passager du Titanic et estimez sa probabilité de survie.")

with st.spinner(
    "Connexion au modèle... Le premier démarrage peut prendre jusqu'à une minute."
):
    api_ready = wake_up_api()

if api_ready:
    st.success("🟢 Modèle disponible")
else:
    st.warning(
        "🟠 Le modèle est encore en cours de démarrage. "
        "Vous pouvez remplir le formulaire et réessayer dans quelques instants."
    )

st.sidebar.header("Profil du passager")

pclass = st.sidebar.selectbox(
    "Classe",
    options=[1, 2, 3],
    index=2,
)

title = st.sidebar.selectbox(
    "Titre",
    options=["Mr", "Mrs", "Miss", "Master"],
)

sex = st.sidebar.selectbox(
    "Sexe",
    options=["male", "female"],
)

age = st.sidebar.slider(
    "Âge",
    min_value=0,
    max_value=100,
    value=28,
)

travel_group_size = st.sidebar.slider(
    "Nombre de personnes voyageant ensemble",
    min_value=1,
    max_value=20,
    value=1,
)

has_cabin = st.sidebar.checkbox(
    "Cabine connue",
    value=False,
)

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

if st.button("Prédire la survie", width="stretch"):
    try:
        # -----------------------------------------------------
        # Prediction
        # -----------------------------------------------------

        with st.spinner(
            "Calcul de la prédiction... "
            "Le serveur peut mettre quelques secondes à répondre."
        ):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=90,
            )

            response.raise_for_status()
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

        # -----------------------------------------------------
        # SHAP explanation
        # -----------------------------------------------------

        with st.spinner("Calcul de l'explication SHAP..."):
            explain_response = requests.post(
                f"{API_URL}/explain",
                json=payload,
                timeout=90,
            )

            explain_response.raise_for_status()
            explanation = explain_response.json()

        factors = explanation["top_factors"]

        st.subheader("Explication de la prédiction")

        factors_df = pd.DataFrame(factors)
        factors_df["abs_contribution"] = factors_df["contribution"].abs()

        factors_df = factors_df.sort_values(
            "abs_contribution",
            ascending=False,
        ).reset_index(drop=True)

        st.dataframe(
            factors_df[
                [
                    "feature",
                    "contribution",
                    "direction",
                ]
            ],
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

        base_value = (
            explanation["survival_probability"] - factors_df["contribution"].sum()
        )

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

    except requests.exceptions.Timeout:
        st.warning(
            "Le serveur met plus de temps que prévu à répondre. "
            "Patientez quelques secondes puis réessayez."
        )

    except requests.exceptions.HTTPError:
        st.error(
            "Le service a répondu avec une erreur. "
            "Veuillez réessayer dans quelques instants."
        )

    except requests.exceptions.RequestException:
        st.error("Impossible de contacter le service de prédiction pour le moment.")
