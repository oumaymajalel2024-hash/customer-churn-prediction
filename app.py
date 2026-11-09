import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

# =====================================================================
# STYLE (CSS personnalisé)
# =====================================================================
CSS = """
<style>
    /* Nettoyage de l'interface Streamlit */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    /* Bandeau d'en-tête */
    .hero {
        background: linear-gradient(120deg, #14232b 0%, #1f6f6b 100%);
        color: #ffffff;
        padding: 2rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; color: #ffffff; padding: 0; }
    .hero p { margin: 0.5rem 0 0 0; font-size: 1.05rem; opacity: 0.9; max-width: 60ch; }

    /* Cartes */
    .card {
        background: #ffffff;
        border: 1px solid #d5dfdd;
        border-radius: 14px;
        padding: 1.5rem;
    }
    .card h3 { margin: 0 0 0.3rem 0; font-size: 1.1rem; color: #14232b; }
    .muted { color: #55676e; font-size: 0.95rem; }

    /* Résultat */
    .big-number { font-size: 3.6rem; font-weight: 800; line-height: 1; margin: 0.4rem 0; }
    .badge {
        display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
        color: #ffffff; font-weight: 700; font-size: 0.95rem;
    }

    /* Jauge de risque : 3 zones + curseur */
    .gauge { position: relative; height: 14px; border-radius: 999px; margin: 1.6rem 0 0.4rem 0;
        background: linear-gradient(90deg,
            #1f6f6b 0%, #1f6f6b 40%,
            #e0a100 40%, #e0a100 60%,
            #e4572e 60%, #e4572e 100%); }
    .gauge .pin {
        position: absolute; top: -6px; width: 6px; height: 26px; border-radius: 3px;
        background: #14232b; transform: translateX(-50%); border: 2px solid #ffffff;
    }
    .gauge-labels { display: flex; justify-content: space-between; font-size: 0.8rem; color: #55676e; }

    /* Liste des leviers */
    .tips { margin: 0.6rem 0 0 0; padding-left: 1.1rem; }
    .tips li { margin-bottom: 0.4rem; }

    /* Titres des sections du formulaire */
    div[data-testid="stForm"] { border: 1px solid #d5dfdd; border-radius: 14px; padding: 1.2rem 1.4rem; background: #ffffff; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button { border-radius: 10px; font-weight: 700; padding: 0.6rem 1.2rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =====================================================================
# MODÈLE
# =====================================================================
@st.cache_resource
def charger():
    modele = joblib.load("models/best_model.joblib")
    colonnes = joblib.load("models/columns.joblib")
    return modele, colonnes


model, columns = charger()


# =====================================================================
# BARRE LATÉRALE
# =====================================================================
with st.sidebar:
    st.markdown("### 📊 À propos du projet")
    st.write(
        "Prédit si un client d'un opérateur télécom va résilier son abonnement, "
        "pour cibler les actions de rétention."
    )
    st.divider()
    st.markdown("**Modèle utilisé**")
    st.write("Random Forest")
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", "0.843")
    c2.metric("Recall", "0.786")
    st.divider()
    st.markdown("**Données**")
    c3, c4 = st.columns(2)
    c3.metric("Clients", "7 043")
    c4.metric("Churn", "26,5 %")
    st.divider()
    st.caption("Dataset : Telco Customer Churn")


# =====================================================================
# EN-TÊTE
# =====================================================================
st.markdown(
    """
    <div class="hero">
        <h1>📉 Prédiction du départ d'un client</h1>
        <p>Renseignez le profil d'un client pour estimer sa probabilité de résiliation
        et obtenir des pistes de rétention.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 1], gap="large")


# =====================================================================
# FORMULAIRE (colonne de gauche)
# =====================================================================
with left:
    with st.form("formulaire_client"):
        tab1, tab2, tab3 = st.tabs(["👤 Profil", "💳 Contrat", "📡 Services"])

        with tab1:
            a, b = st.columns(2)
            with a:
                gender = st.selectbox("Genre", ["Female", "Male"])
                partner = st.selectbox("En couple", ["Yes", "No"])
            with b:
                senior = st.selectbox(
                    "Senior (65 ans et +)", [0, 1],
                    format_func=lambda x: "Oui" if x else "Non",
                )
                dependents = st.selectbox("Personnes à charge", ["Yes", "No"])
            tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

        with tab2:
            contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Mode de paiement", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            a, b = st.columns(2)
            with a:
                paperless = st.selectbox("Facture sans papier", ["Yes", "No"])
            with b:
                monthly = st.number_input("Facture mensuelle (€)", 15.0, 125.0, 70.0, step=0.5)

        with tab3:
            a, b, c = st.columns(3)
            with a:
                phone = st.selectbox("Téléphone", ["Yes", "No"])
                multiple = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"])
                internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
            with b:
                security = st.selectbox("Sécurité en ligne", ["No", "Yes", "No internet service"])
                backup = st.selectbox("Sauvegarde en ligne", ["No", "Yes", "No internet service"])
                protection = st.selectbox("Protection appareil", ["No", "Yes", "No internet service"])
            with c:
                support = st.selectbox("Support technique", ["No", "Yes", "No internet service"])
                tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                movies = st.selectbox("Streaming films", ["No", "Yes", "No internet service"])

        envoyer = st.form_submit_button("Analyser ce client", type="primary", use_container_width=True)


# =====================================================================
# PRÉPARATION (mêmes transformations que dans le notebook, inchangé)
# =====================================================================
def preparer():
    total = monthly * tenure
    ligne = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    df = pd.DataFrame([ligne])

    # Feature engineering
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12 mois", "13-24 mois", "25-48 mois", "49-72 mois"]).astype(str)
    services = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["nb_services"] = (df[services] == "Yes").sum(axis=1)
    df["avg_charges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["high_risk"] = ((df["Contract"] == "Month-to-month") &
                       (df["PaymentMethod"] == "Electronic check")).astype(int)

    # Encodage, puis alignement exact sur les colonnes du modèle
    df = pd.get_dummies(df, dtype=int)
    return df.reindex(columns=columns, fill_value=0)


def leviers():
    """Pistes de rétention, basées sur les tendances observées dans l'EDA."""
    liste = []
    if contract == "Month-to-month":
        liste.append("Proposer un contrat de 1 ou 2 ans avec une remise.")
    if payment == "Electronic check":
        liste.append("Encourager le passage au prélèvement automatique.")
    if tenure <= 12:
        liste.append("Client récent : renforcer l'accompagnement les premiers mois.")
    if internet == "Fiber optic":
        liste.append("Vérifier la satisfaction sur la fibre (churn plus élevé dans les données).")
    if internet != "No" and support == "No":
        liste.append("Proposer l'option support technique.")
    return liste


# =====================================================================
# RÉSULTAT (colonne de droite)
# =====================================================================
with right:
    if not envoyer:
        st.markdown(
            """
            <div class="card">
                <h3>Résultat</h3>
                <p class="muted">Remplissez le formulaire puis cliquez sur
                <b>Analyser ce client</b>. La probabilité de départ et les pistes de
                rétention s'afficheront ici.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        proba = float(model.predict_proba(preparer())[0, 1])
        pct = proba * 100

        if proba >= 0.6:
            couleur, niveau, message = "#e4572e", "Risque élevé", "Ce client risque de partir. Une action de rétention est recommandée."
        elif proba >= 0.4:
            couleur, niveau, message = "#e0a100", "Risque moyen", "Client à surveiller."
        else:
            couleur, niveau, message = "#1f6f6b", "Risque faible", "Ce client devrait rester."

        pistes = leviers()
        if pistes:
            html_pistes = "<ul class='tips'>" + "".join(f"<li>{p}</li>" for p in pistes) + "</ul>"
        else:
            html_pistes = "<p class='muted'>Aucun levier évident pour ce profil.</p>"

        st.markdown(
            f"""
            <div class="card">
                <h3>Probabilité de départ</h3>
                <div class="big-number" style="color:{couleur}">{pct:.1f} %</div>
                <span class="badge" style="background:{couleur}">{niveau}</span>
                <p class="muted" style="margin-top:0.8rem">{message}</p>

                <div class="gauge"><div class="pin" style="left:{min(max(pct, 1), 99):.1f}%"></div></div>
                <div class="gauge-labels"><span>0 %</span><span>40 %</span><span>60 %</span><span>100 %</span></div>
            </div>

            <div class="card" style="margin-top:1rem">
                <h3>Leviers de rétention</h3>
                {html_pistes}
            </div>
            """,
            unsafe_allow_html=True,
        )