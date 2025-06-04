import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
try:
    df = pd.read_csv("BAHAN BAKAR MOBIL 2023.csv", encoding="latin1")
except:
    df = pd.read_csv("BAHAN BAKAR MOBIL 2023.csv", encoding="ISO-8859-1")

st.title("📊 Regresi Linear Berganda (2 Variabel X)")

  # --- Tampilkan Data Awal ---
st.subheader("📋 Data Awal")
st.dataframe(df)

# --- Pilih Variabel X dan Y ---
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

x_vars = st.multiselect("Pilih 2 variabel X (independen):", numeric_cols, default=["Comb (mpg)", "CO2 Emissions (g/km)"])
y_var = st.selectbox("Pilih variabel Y (dependen):", numeric_cols)

# --- Validasi Jumlah X ---
if len(x_vars) != 2:
    st.warning("⚠️ Pilih tepat 2 variabel X untuk regresi linear berganda.")
else:
    # --- Tampilkan Data Awal ---
    st.subheader("📋 Data Awal")
    selected_cols = x_vars + [y_var]
    preview_df = df[selected_cols].dropna()
    st.dataframe(preview_df)

    # --- Model Regresi ---
    st.subheader("🔍 Hasil Regresi Linear")
    X = preview_df[x_vars]
    y = preview_df[y_var]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    st.markdown(f"""
    **Koefisien:**
    - {x_vars[0]} = {model.coef_[0]:.2f}
    - {x_vars[1]} = {model.coef_[1]:.2f}

    **Intercept:** {model.intercept_:.2f}  
    **R² Score:** {r2_score(y, y_pred):.3f}  
    **RMSE:** {np.sqrt(mean_squared_error(y, y_pred)):.2f}
    """)

    # --- Visualisasi Prediksi vs Aktual ---
    st.subheader("📈 Grafik Prediksi vs Aktual")
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, color='green', alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
    ax.set_title(f"Prediksi vs Aktual untuk {y_var}")
    st.pyplot(fig)
