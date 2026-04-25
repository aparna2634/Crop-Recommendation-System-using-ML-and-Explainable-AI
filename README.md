# Crop-Recommendation-System-using-ML-and-Explainable-AI

## Overview
A Machine Learning web app that recommends the best crop to grow based on soil and climate inputs. The farmer enters 7 simple readings — the system instantly predicts the most suitable crop, shows a confidence score, and explains *why* that crop was chosen using AI.

Built as a complete end-to-end pipeline: from raw data → preprocessing → model training → explainability → deployed web app.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn (Random Forest, MLP/ANN) |
| Explainability | SHAP, LIME |
| Model Saving | Joblib |
| Web App | Streamlit |
| Dataset | [Kaggle — Crop Recommendation](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |

---

## App Features

- **7 input sliders** — enter Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall
- **Model selector** — switch between Random Forest and ANN predictions
- **Crop prediction card** — displays the recommended crop with an emoji and a growing tip
- **Confidence score & gauge** — shows how certain the model is (e.g., 99.1%)
- **Top-5 candidates** — see the next best crop options with their probabilities
- **SHAP chart** — explains which features influenced the prediction globally
- **LIME chart** — explains why your specific input led to that prediction
- **EDA plot gallery** — view all training charts (distributions, heatmaps, confusion matrix)

---

## Conclusion

The Crop Recommendation System predicts the most suitable crop with ~99% accuracy using Random Forest and ~98% using an Artificial Neural Network. By combining both models with SHAP and LIME explainability, the app doesn't just predict — it explains. This makes it practical and trustworthy for real-world use, helping farmers make data-driven decisions without needing agronomic expertise.
