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


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("🚢 Titanic Survival Predictor")

st.write(
    "Imaginez que vous embarquez à bord du Titanic. "
    "Renseignez votre profil pour estimer vos chances de survie."
)


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


# ---------------------------------------------------------------------
# Passenger form
# ---------------------------------------------------------------------

st.subheader("Votre profil")


pclass = st.selectbox(
    "Classe de voyage",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "1ère classe",
        2: "2ème classe",
        3: "3ème classe",
    }[x],
    index=2,
)


sex_fr = st.selectbox(
    "Sexe",
    options=["Masculin", "Féminin"],
)

sex = {
    "Masculin": "male",
    "Féminin": "female",
}[sex_fr]


age = st.number_input(
    "Âge",
    min_value=0,
    max_value=100,
    value=28,
    step=1,
)


travel_group_size = st.number_input(
    "Nombre de personnes voyageant ensemble",
    min_value=1,
    max_value=20,
    value=1,
    step=1,
)


# ---------------------------------------------------------------------
# Derive Title
# ---------------------------------------------------------------------

if sex == "male":
    title = "Master" if age < 18 else "Mr"

else:
    if age < 18:
        title = "Miss"

    else:
        female_title = st.selectbox(
            "Titre",
            options=["Mlle", "Mme"],
            help=(
                "Cette information est utilisée par le modèle "
                "car le titre était fortement associé à la survie."
            ),
        )

        title = {
            "Mlle": "Miss",
            "Mme": "Mrs",
        }[female_title]


# Hidden/default model features
has_cabin = False


payload = {
    "pclass": pclass,
    "title": title,
    "age": age,
    "sex": sex,
    "travel_group_size": travel_group_size,
    "has_cabin": has_cabin,
}


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

if st.button(
    "Estimer mes chances de survie",
    width="stretch",
    type="primary",
):
    try:
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

        st.divider()
        st.subheader("Résultat")

        st.metric(
            label="Probabilité estimée de survie",
            value=f"{probability:.1%}",
        )

        st.progress(probability)

        if result["prediction"] == 1:
            st.success("Le modèle estime que vous auriez probablement survécu.")
        else:
            st.error("Le modèle estime que vous n'auriez probablement pas survécu.")

        # -------------------------------------------------------------
        # SHAP
        # -------------------------------------------------------------

        with st.spinner("Analyse des facteurs ayant influencé la prédiction..."):
            explain_response = requests.post(
                f"{API_URL}/explain",
                json=payload,
                timeout=90,
            )

            explain_response.raise_for_status()
            explanation = explain_response.json()

        factors = explanation["top_factors"]

        st.divider()
        st.subheader("Pourquoi cette prédiction ?")

        st.write(
            "Le graphique ci-dessous montre les variables qui ont poussé "
            "le modèle vers une probabilité de survie plus élevée ou plus faible."
        )

        factors_df = pd.DataFrame(factors)
        factors_df["abs_contribution"] = factors_df["contribution"].abs()

        factors_df = factors_df.sort_values(
            "abs_contribution",
            ascending=False,
        ).reset_index(drop=True)

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
