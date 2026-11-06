# Customer Churn Prediction

Projet de Machine Learning qui prédit si un client d'un opérateur télécom va résilier son abonnement (churn), avec une application web interactive.

## Objectif

Identifier les clients à risque de départ pour permettre des actions de rétention ciblées.

## Dataset

Telco Customer Churn : 7043 clients, 21 colonnes (profil, services souscrits, contrat, facturation). Taux de churn : 26,5 %.

## Structure du projet

```
customer-churn-prediction/
├── data/
│   ├── raw/            # données brutes (database.csv)
│   └── processed/      # données nettoyées et jeux train/test
├── notebooks/
│   └── 04_machine_learning.ipynb
├── models/             # modèle sauvegardé (best_model.joblib)
├── app.py              # application Streamlit
├── requirements.txt
└── README.md
```

## Méthodologie

1. **Nettoyage** : conversion de `TotalCharges` en numérique, 11 valeurs vides (clients avec ancienneté 0) remplacées par 0, aucun doublon.
2. **Analyse exploratoire (EDA)** : le churn est plus élevé pour les contrats mensuels, la fibre optique, le paiement par chèque électronique et les clients récents.
3. **Feature engineering** : `tenure_group`, `nb_services`, `avg_charges`, `high_risk`.
4. **Modélisation** : Logistic Regression, Decision Tree, Random Forest, XGBoost, avec gestion du déséquilibre des classes.
5. **Explicabilité** : analyse SHAP sur XGBoost.

## Résultats

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.758 | 0.530 | 0.786 | 0.633 | 0.843 |
| Logistic Regression | 0.735 | 0.501 | 0.797 | 0.615 | 0.842 |
| XGBoost | 0.749 | 0.519 | 0.783 | 0.624 | 0.839 |
| Decision Tree | 0.722 | 0.485 | 0.762 | 0.593 | 0.815 |

Le **Random Forest** a été retenu (meilleur ROC-AUC et meilleur équilibre global). Les trois meilleurs modèles sont très proches (moins de 0,005 d'écart de ROC-AUC).

## Installation et lancement

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Application Streamlit

L'application permet de saisir le profil d'un client et affiche sa probabilité de départ avec un niveau de risque (faible, moyen, élevé).

## Technologies

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, matplotlib, seaborn.

## Équipe

- Personne 1 : Data Analysis (nettoyage, EDA, feature engineering)
- Personne 2 : Machine Learning (modèles, évaluation, SHAP)
- Travail commun : application Streamlit, README, rapport et présentation