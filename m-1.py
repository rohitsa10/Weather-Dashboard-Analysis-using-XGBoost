import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

st.title("🌧️ City-wise Rain Prediction with ML Insights")

uploaded_file = st.file_uploader("📂 Upload your weather dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Create binary target column 'Rain' if not present
    if "Rain" not in df.columns:
        df["Rain"] = np.where(
            (df["Humidity"] > 75)
            | (df["Cloudiness (%)"] > 60)
            | (df["Weather"].str.contains("rain", case=False, na=False)),
            1,
            0,
        )
        st.info("💡 'Rain' column auto-created based on humidity, cloudiness, and weather text.")

    # Convert datetime columns
    if "Sunrise (UTC)" in df.columns:
        df["Sunrise_hour"] = pd.to_datetime(df["Sunrise (UTC)"], errors="coerce").dt.hour
    if "Sunset (UTC)" in df.columns:
        df["Sunset_hour"] = pd.to_datetime(df["Sunset (UTC)"], errors="coerce").dt.hour

    # Drop unnecessary/non-numeric columns
    drop_cols = ["Sunrise (UTC)", "Sunset (UTC)", "Timezone", "Weather"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Define features and target
    features = ["Humidity", "Pressure", "Wind Speed", "Cloudiness (%)", "Feels Like", "Temperature"]
    features = [col for col in features if col in df.columns]
    X = df[features]
    y = df["Rain"]

    # Split and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained successfully! Accuracy: {acc * 100:.2f}%")

    # Classification report
    st.text("📋 Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Select multiple cities
    if "City" in df.columns:
        cities = st.multiselect("🏙️ Select up to 5 cities for rain prediction", df["City"].unique().tolist(), max_selections=5)

        if cities:
            selected_df = df[df["City"].isin(cities)]
            city_preds = model.predict(selected_df[features])
            selected_df["Predicted_Rain"] = city_preds

            st.subheader("🌦️ City-wise Rain Prediction Results")
            result_df = selected_df[["City"] + features + ["Predicted_Rain"]].copy()
            result_df["Predicted_Rain"] = result_df["Predicted_Rain"].map({1: "🌧️ Rain", 0: "☀️ No Rain"})
            st.dataframe(result_df)

            # Insights
            st.subheader("📈 Insights")
            avg = selected_df.groupby("City")[features].mean().round(2)
            st.write("Average Weather Parameters for Selected Cities:")
            st.dataframe(avg)

            fig, ax = plt.subplots(figsize=(8, 5))
            avg.plot(kind="bar", ax=ax)
            plt.title("Average Weather Conditions by City")
            plt.ylabel("Value")
            plt.xticks(rotation=30)
            st.pyplot(fig)

    else:
        st.warning("⚠️ 'City' column not found in dataset — please include it to enable city selection.")

else:
    st.info("📥 Upload your weather dataset (CSV) to begin.")
