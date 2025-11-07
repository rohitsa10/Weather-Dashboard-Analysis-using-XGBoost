import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import numpy as np

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="🌦️ City Weather Analysis Dashboard", layout="wide")

plt.style.use('dark_background')
sns.set_theme(style="darkgrid")
sns.set_context("talk", font_scale=1)
custom_color = "#5dade2"

# -----------------------------
# Sidebar Menu
# -----------------------------
menu = [
    "Overview",
    "Temperature & Humidity Trends",
    "Pressure & Wind Speed Relation",
    "Cloudiness & Visibility",
    "Weather Condition Frequency",
    "Comfort Index & Anomaly Detection",
    "Weather Severity & Correlation Heatmap",
    "City Comparison Dashboard"
]
choice = st.sidebar.selectbox("📊 Select Analysis Section", menu)

# -----------------------------
# Upload CSV
# -----------------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your weather CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    # -----------------------------
    # City Selection
    # -----------------------------
    city_list = df['City'].unique().tolist()
    selected_cities = st.sidebar.multiselect("🏙️ Select up to 5 cities:", city_list, default=city_list[:5])
    if len(selected_cities) == 0:
        st.warning("⚠️ Please select at least one city.")
        st.stop()
    elif len(selected_cities) > 5:
        st.warning("⚠️ Please select only up to 5 cities.")
        st.stop()

    df = df[df['City'].isin(selected_cities)]

    # -----------------------------
    # SECTION 1 — OVERVIEW
    # -----------------------------
    if choice == "Overview":
        st.title("🌍 City Weather Data Overview")
        st.write(f"Analyzing: **{', '.join(selected_cities)}**")
        st.dataframe(df.head())
        st.subheader("📈 Summary Statistics")
        st.dataframe(df.describe())

    # -----------------------------
    # SECTION 2 — Temperature & Humidity Trends
    # -----------------------------
    elif choice == "Temperature & Humidity Trends":
        st.title("🌡️ Temperature & Humidity Trends")
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(8,5))
            sns.barplot(data=df, x='City', y='Temperature', color=custom_color)
            plt.title("Average Temperature by City", fontsize=16, color='white', weight='bold')
            plt.xlabel("")
            plt.ylabel("Temperature (°C)")
            st.pyplot(plt)
        with col2:
            plt.figure(figsize=(8,5))
            sns.lineplot(data=df, x='City', y='Humidity', marker='o', color=custom_color)
            plt.title("Average Humidity by City", fontsize=16, color='white', weight='bold')
            plt.xlabel("")
            plt.ylabel("Humidity (%)")
            st.pyplot(plt)
        st.markdown("💡 **Insight:** Warmer cities show lower humidity, cooler ones have higher moisture.")

        # -----------------------------
    # SECTION 3 — Pressure & Wind Speed Relation (Improved)
    # -----------------------------
    elif choice == "Pressure & Wind Speed Relation":
        st.title("🌬️ Pressure vs Wind Speed by City")

        col1, col2 = st.columns(2)

        with col1:
            plt.figure(figsize=(8,5))
            sns.barplot(data=df, x='City', y='Pressure', hue='City', palette='cool', legend=False)
            plt.title("Average Pressure by City", fontsize=13, fontweight='bold')
            plt.xlabel("City", fontsize=11)
            plt.ylabel("Pressure (hPa)", fontsize=11)
            plt.xticks(rotation=30, fontsize=10, color='black')
            plt.yticks(fontsize=10, color='black')
            plt.tight_layout()
            st.pyplot(plt)

        with col2:
            plt.figure(figsize=(8,5))
            sns.lineplot(data=df, x='City', y='Wind Speed', marker='o', color='#5dade2', linewidth=3)
            plt.title("Wind Speed Trend by City", fontsize=13, fontweight='bold')
            plt.xlabel("City", fontsize=11)
            plt.ylabel("Wind Speed (km/h)", fontsize=11)
            plt.xticks(rotation=30, fontsize=10, color='black')
            plt.yticks(fontsize=10, color='black')
            plt.tight_layout()
            st.pyplot(plt)

        st.markdown("💡 **Insight:** Cities with lower pressure often experience stronger wind activity.")


    # -----------------------------
    # SECTION 4 — Cloudiness & Visibility (Improved)
    # -----------------------------
    elif choice == "Cloudiness & Visibility":
        st.title("☁️ Cloudiness & Visibility Comparison")

        col1, col2 = st.columns(2)

        with col1:
            plt.figure(figsize=(8,5))
            sns.barplot(data=df, x='City', y='Cloudiness (%)', hue='City', palette='Blues', legend=False)
            plt.title("Average Cloudiness by City", fontsize=13, fontweight='bold')
            plt.xlabel("City", fontsize=11)
            plt.ylabel("Cloudiness (%)", fontsize=11)
            plt.xticks(rotation=30, fontsize=10, color='black')
            plt.yticks(fontsize=10, color='black')
            plt.tight_layout()
            st.pyplot(plt)

        with col2:
            plt.figure(figsize=(8,5))
            sns.lineplot(data=df, x='City', y='Visibility (km)', marker='o', color='#58d68d', linewidth=3)
            plt.title("Average Visibility by City", fontsize=13, fontweight='bold')
            plt.xlabel("City", fontsize=11)
            plt.ylabel("Visibility (km)", fontsize=11)
            plt.xticks(rotation=30, fontsize=10, color='black')
            plt.yticks(fontsize=10, color='black')
            plt.tight_layout()
            st.pyplot(plt)

        st.markdown("💡 **Insight:** Heavier cloud cover usually reduces visibility across cities.")



    # -----------------------------
    # SECTION 5 — Weather Condition Frequency
    # -----------------------------
    elif choice == "Weather Condition Frequency":
        st.title("🌦️ Weather Condition Frequency")
        weather_counts = df['Weather'].value_counts()
        plt.figure(figsize=(8,5))
        sns.barplot(x=weather_counts.index, y=weather_counts.values, color=custom_color)
        plt.title("Weather Conditions Across Selected Cities", color='white', weight='bold')
        plt.xticks(rotation=25)
        st.pyplot(plt)
        st.markdown("💡 **Insight:** Overcast and scattered clouds dominate most observations.")

    # -----------------------------
    # SECTION 6 — Comfort Index
    # -----------------------------
    elif choice == "Comfort Index & Anomaly Detection":
        st.title("😌 Comfort Index & Anomaly Detection")
        df['Comfort Index'] = (100 - abs(df['Temperature'] - df['Feels Like'])) + (100 - df['Humidity']) / 2
        plt.figure(figsize=(8,5))
        sns.barplot(data=df, x='City', y='Comfort Index', color=custom_color)
        plt.title("Comfort Index by City", color='white', weight='bold')
        st.pyplot(plt)
        st.markdown("💡 **Insight:** Higher index = more pleasant and stable weather.")

    # -----------------------------
    # SECTION 7 — Severity & Correlation
    # -----------------------------
    elif choice == "Weather Severity & Correlation Heatmap":
        st.title("🔥 Weather Severity & Correlation Heatmap")
        df['Severity Index'] = (df['Wind Speed'] * 0.3) + (df['Cloudiness (%)'] * 0.4) + (100 - df['Visibility (km)'])
        plt.figure(figsize=(8,5))
        sns.barplot(data=df, x='City', y='Severity Index', color=custom_color)
        plt.title("Weather Severity Index by City", color='white', weight='bold')
        st.pyplot(plt)
        st.subheader("📊 Correlation Heatmap")
        plt.figure(figsize=(8,6))
        sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
        plt.title("Feature Correlation", color='white', weight='bold')
        st.pyplot(plt)

    # -----------------------------
    # SECTION 8 — City Comparison
    # -----------------------------
    elif choice == "City Comparison Dashboard":
        st.title("🏙️ City Comparison Dashboard")
        numeric_cols = df.select_dtypes(include=['number']).columns
        city1 = st.selectbox("Select City 1", selected_cities, key='c1')
        city2 = st.selectbox("Select City 2", selected_cities, key='c2')
        city_compare = df[df['City'].isin([city1, city2])]
        st.dataframe(city_compare[numeric_cols].mean().to_frame(name="Average").T)
        plt.figure(figsize=(8,5))
        sns.barplot(data=city_compare, x='City', y='Temperature', color=custom_color)
        plt.title(f"Temperature Comparison: {city1} vs {city2}", color='white', weight='bold')
        st.pyplot(plt)

    # -----------------------------

else:
    st.warning("📥 Please upload a CSV file to begin the analysis.")
