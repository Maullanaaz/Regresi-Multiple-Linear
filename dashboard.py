
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
# Gaya font global
st.markdown("""
    <style>
    html, body, [class*='css']  {
        font-family: 'Poppins', sans-serif;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


# Styling global
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
try:
    df = pd.read_csv("BAHAN BAKAR MOBIL 2023.csv", encoding="latin1")
except:
    df = pd.read_csv("BAHAN BAKAR MOBIL 2023.csv", encoding="ISO-8859-1")

# Judul utama
st.title("📊 Prediksi Konsumsi Bahan Bakar Mobil dengan Regresi Linear Berganda")

# --- Tampilkan Data Awal ---
st.markdown("### 📘 Data Awal Kendaraan")
st.dataframe(df)

# --- Pilih Variabel X dan Y ---
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
x_vars = ["Comb (mpg)", "CO2 Emissions (g/km)"]
y_var = ["Fuel Consumption (L/100Km)"]

# --- Tampilkan Data yang Digunakan untuk Regresi ---
st.markdown("### 📊 Data yang Digunakan untuk Model Regresi")
selected_cols = x_vars + y_var
preview_df = df[selected_cols].dropna()
st.dataframe(preview_df)

# --- Model Regresi ---
st.markdown("### 📈 Hasil Regresi Linear")
X = preview_df[x_vars]
y = preview_df[y_var[0]]
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r_squared = r2_score(y, y_pred)

st.markdown(f"""
**Koefisien Model:**
- {x_vars[0]} = {model.coef_[0]:.2f}
- {x_vars[1]} = {model.coef_[1]:.2f}

**Evaluasi Model:**
- MAE = {mae:.2f}  
- MSE = {mse:.2f}  
- R² Score = {r_squared:.3f}
""")

# --- Visualisasi Prediksi vs Aktual dengan Warna Berdasarkan Index ---
st.markdown("### 🎨 Grafik Prediksi vs Aktual berdasarkan ID")

preview_df = preview_df.copy()
preview_df["y_pred"] = y_pred
preview_df["index"] = range(len(preview_df))

def get_color(idx):
    if idx <= 92:
        return "saddlebrown"
    elif idx <= 184:
        return "orange"
    elif idx <= 276:
        return "blue"
    elif idx <= 368:
        return "green"
    elif idx <= 460:
        return "gray"
    elif idx <= 552:
        return "red"
    elif idx <= 644:
        return "gold"
    else:
        return "purple"

preview_df["warna"] = preview_df["index"].apply(get_color)

fig, ax = plt.subplots()
for _, row in preview_df.iterrows():
    ax.scatter(row[y_var[0]], row["y_pred"], color=row["warna"], alpha=0.6, s=80)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax.set_xlabel("Aktual")
ax.set_ylabel("Prediksi")
ax.set_title(f"Prediksi vs Aktual untuk {y_var[0]}")
st.pyplot(fig)

st.markdown("#### 📁 Keterangan Warna Berdasarkan ID")
st.markdown("""
- 🟤 Cokelat: ID 0–92  
- 🟠 Oranye: ID 93–184  
- 🔵 Biru: ID 185–276  
- 🟢 Hijau: ID 277–368  
- ⚫ Abu-Abu: ID 369–460  
- 🔴 Merah: ID 461–552  
- 🟡 Kuning: ID 553–644  
- 🟣 Ungu: ID 645–832
""")

# === Insight Tambahan: Mobil Paling Hemat Asia vs Eropa ===
st.markdown("---")
st.markdown("### 🌍 Insight Tambahan: Mobil Terhemat Asia vs Eropa")

fuel_mapping = {'Z': 'Gasoline', 'X': 'Hybrid', 'D': 'Diesel', 'E': 'Electric'}
df['Fuel Type Name'] = df['Fuel Type'].map(fuel_mapping).fillna('Unknown')

asia_makes = ['Acura', 'Genesis', 'Honda', 'Hyundai', 'Infiniti', 'Kia', 'Lexus',
              'Mazda', 'Mitsubishi', 'Nissan', 'Subaru', 'Toyota']
eropa_makes = ['Alfa Romeo', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Bugatti',
               'FIAT', 'Jaguar', 'Land Rover', 'Maserati', 'Mercedes-Benz',
               'MINI', 'Porsche', 'Rolls-Royce', 'Volkswagen', 'Volvo']

df['Region'] = df['Make'].apply(lambda x: 'Asia' if x in asia_makes else ('Eropa' if x in eropa_makes else 'Other'))

df_filtered = df[df['Fuel Type Name'].isin(['Gasoline', 'Hybrid', 'Diesel', 'Electric'])]

top_10 = df_filtered.groupby(['Region', 'Fuel Type Name'], group_keys=False).apply(
    lambda x: x.sort_values(by='Comb (L/100 km)').head(10)
)

top_10_asia = top_10[top_10['Region'] == 'Asia']
top_10_eropa = top_10[top_10['Region'] == 'Eropa']

st.markdown("### 🚗 Top 10 Mobil Terhemat - Asia")
st.dataframe(top_10_asia[['Make', 'Model', 'Fuel Type Name', 'Comb (L/100 km)']].reset_index(drop=True))

st.markdown("### 🚗 Top 10 Mobil Terhemat - Eropa")
st.dataframe(top_10_eropa[['Make', 'Model', 'Fuel Type Name', 'Comb (L/100 km)']].reset_index(drop=True))

most_asia = top_10_asia.sort_values(by='Comb (L/100 km)').head(1).iloc[0]
most_eropa = top_10_eropa.sort_values(by='Comb (L/100 km)').head(1).iloc[0]

st.markdown("**Mobil Terhemat dari Asia:**")
st.success(f"{most_asia['Make']} {most_asia['Model']} ({most_asia['Fuel Type Name']}) - {most_asia['Comb (L/100 km)']} L/100 km")

st.markdown("**Mobil Terhemat dari Eropa:**")
st.success(f"{most_eropa['Make']} {most_eropa['Model']} ({most_eropa['Fuel Type Name']}) - {most_eropa['Comb (L/100 km)']} L/100 km")

st.markdown("**🔎 Mobil Paling Hemat di Antara Keduanya:**")
if most_asia['Comb (L/100 km)'] < most_eropa['Comb (L/100 km)']:
    st.info(f"🏆 {most_asia['Make']} {most_asia['Model']} dari Asia lebih hemat.")
elif most_asia['Comb (L/100 km)'] > most_eropa['Comb (L/100 km)']:
    st.info(f"🏆 {most_eropa['Make']} {most_eropa['Model']} dari Eropa lebih hemat.")
else:
    st.info("Keduanya memiliki efisiensi bahan bakar yang sama.")
