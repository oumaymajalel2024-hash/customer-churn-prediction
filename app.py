import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")

@st.cache_resource
def charger():
    modele = joblib.load("models/best_model.joblib")
    colonnes = joblib.load("models/columns.joblib")
    return modele, colonnes

model, columns = charger()

st.title("📉 Prédiction du départ d'un client")
st.write("Renseigne les informations du client pour estimer sa probabilité de départ (churn).")

# ---------- Formulaire ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Profil")
    gender = st.selectbox("Genre", ["Female", "Male"])
    senior = st.selectbox("Senior (65 ans et +)", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    partner = st.selectbox("En couple", ["Yes", "No"])
    dependents = st.selectbox("Personnes à charge", ["Yes", "No"])
    tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

with col2:
    st.subheader("Contrat et facturation")
    contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Facture sans papier", ["Yes", "No"])
    payment = st.selectbox("Mode de paiement", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Facture mensuelle", 15.0, 125.0, 70.0, step=0.5)

st.subheader("Services")
c1, c2, c3 = st.columns(3)
with c1:
    phone = st.selectbox("Service téléphone", ["Yes", "No"])
    multiple = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
with c2:
    security = st.selectbox("Sécurité en ligne", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Sauvegarde en ligne", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Protection appareil", ["No", "Yes", "No internet service"])
with c3:
    support = st.selectbox("Support technique", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming films", ["No", "Yes", "No internet service"])

# ---------- Préparation (mêmes transformations que dans le notebook) ----------
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

# ---------- Prédiction ----------
if st.button("Prédire", type="primary"):
    proba = model.predict_proba(preparer())[0, 1]
    st.metric("Probabilité de départ", f"{proba:.1%}")
    st.progress(float(proba))

    if proba >= 0.6:
        st.error("🔴 Risque ÉLEVÉ : ce client risque de partir. Action de rétention recommandée.")
    elif proba >= 0.4:
        st.warning("🟠 Risque MOYEN : à surveiller.")
    else:
        st.success("🟢 Risque FAIBLE : ce client devrait rester.")