import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
try:
    df = pd.read_csv("BAHAN BAKAR MOBIL 2023.csv", encoding="latin1")
except:
    df = pd.read_csv("BAHAN BAKAR MOBIL 2023.csv", encoding="ISO-8859-1")

st.title("\U0001F4CA Regresi Linear Berganda (2 Variabel X)")

# --- Tampilkan Data Awal ---
st.subheader("\U0001F4CB Data Awal")
st.dataframe(df)

# --- Pilih Variabel X dan Y ---
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

x_vars = ["Comb (mpg)", "CO2 Emissions (g/km)"]
y_var = ["Fuel Consumption (L/100Km)"]

# --- Tampilkan Data yang Digunakan untuk Regresi ---
st.subheader("\U0001F4CB Data yang Digunakan untuk Regresi")
selected_cols = x_vars + y_var
preview_df = df[selected_cols].dropna()
st.dataframe(preview_df)

# --- Model Regresi ---
st.subheader("\U0001F50D Hasil Regresi Linear")
X = preview_df[x_vars]
y = preview_df[y_var[0]]
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r_squared = r2_score(y, y_pred)

st.markdown(f"""
**Koefisien:**
- {x_vars[0]} = {model.coef_[0]:.2f}
- {x_vars[1]} = {model.coef_[1]:.2f}

**Mean Absolute Error (MAE):** {mae:.2f}  
**Mean Squared Error (MSE):** {mse:.2f}  
**R² Score:** {r_squared:.3f}
""")

# --- Visualisasi Prediksi vs Aktual dengan Warna Berdasarkan Index ---
st.subheader("\U0001F4C8 Grafik Prediksi vs Aktual (Berwarna berdasarkan range ID)")

# Tambahkan kolom prediksi dan index ke dataframe
preview_df = preview_df.copy()
preview_df["y_pred"] = y_pred
preview_df["index"] = range(1, len(preview_df) + 1)

# Fungsi pewarnaan berdasarkan index
def get_color(idx):
    if idx <= 75:
        return "purple"
    elif idx <= 150:
        return "orange"
    elif idx <= 225:
        return "blue"
    elif idx <= 300:
        return "green"
    elif idx <= 375:
        return "pink"
    elif idx <= 450:
        return "gray"
    elif idx <= 525:
        return "yellow"
    elif idx <= 600:
        return "brown"
    elif idx <= 675:
        return "teal"
    elif idx <= 750:
        return "red"
    else:
        return "black"

# Tambahkan kolom warna
preview_df["warna"] = preview_df["index"].apply(get_color)

# Buat scatter plot dengan warna sesuai kelompok dan label ID
fig, ax = plt.subplots()
for i, row in preview_df.iterrows():
    ax.scatter(row[y_var[0]], row["y_pred"], color=row["warna"], alpha=0.6, s=20)
    ax.text(row[y_var[0]], row["y_pred"], str(row["index"]), fontsize=4, alpha=0.5)

# Garis referensi
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax.set_xlabel("Aktual")
ax.set_ylabel("Prediksi")
ax.set_title(f"Prediksi vs Aktual untuk {y_var[0]}")
st.pyplot(fig)

# Legend warna manual
st.markdown("#### \U0001F5C2️ Keterangan Warna Berdasarkan ID")
st.markdown("""
- 🔸 **Ungu**: ID 1–75  
- 🔶 **Oranye**: ID 76–150  
- 🔹 **Biru**: ID 151–225  
- 🔵 **Hijau**: ID 226–300  
- 🌸 **Pink**: ID 301–375  
- ⚫ **Abu-abu**: ID 376–450  
- 🟡 **Kuning**: ID 451–525  
- 🟣 **Cokelat**: ID 526–600  
- 🟦 **Toska**: ID 601–675  
- 🔴 **Merah**: ID 676–750  
- ⚫ **Hitam**: ID 751–832
""")

# === Insight Tambahan: Mobil Paling Hemat Asia vs Eropa ===
st.markdown("---")
st.subheader("\U0001F50D Insight Tambahan: Perbandingan Mobil Terhemat Asia vs Eropa")

# Mapping jenis bahan bakar
fuel_mapping = {
    'Z': 'Gasoline',
    'X': 'Hybrid',
    'D': 'Diesel',
    'E': 'Electric'
}
df['Fuel Type Name'] = df['Fuel Type'].map(fuel_mapping).fillna('Unknown')

# Definisi merek Asia dan Eropa
asia_makes = ['Acura', 'Genesis', 'Honda', 'Hyundai', 'Infiniti', 'Kia', 'Lexus',
              'Mazda', 'Mitsubishi', 'Nissan', 'Subaru', 'Toyota']
eropa_makes = ['Alfa Romeo', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Bugatti',
               'FIAT', 'Jaguar', 'Land Rover', 'Maserati', 'Mercedes-Benz',
               'MINI', 'Porsche', 'Rolls-Royce', 'Volkswagen', 'Volvo']

# Tambah kolom region
df['Region'] = df['Make'].apply(lambda x: 'Asia' if x in asia_makes else ('Eropa' if x in eropa_makes else 'Other'))

# Filter bahan bakar valid
df_filtered = df[df['Fuel Type Name'].isin(['Gasoline', 'Hybrid', 'Diesel', 'Electric'])]

# Top 20 mobil paling hemat per region & fuel type
top_10 = df_filtered.groupby(['Region', 'Fuel Type Name'], group_keys=False).apply(
    lambda x: x.sort_values(by='Comb (L/100 km)').head(10)
)

# Top Asia & Eropa
top_10_asia = top_10[top_10['Region'] == 'Asia']
top_10_eropa = top_10[top_10['Region'] == 'Eropa']

# Tampilkan Top 10 Mobil Asia dan Eropa
st.markdown("---")
st.subheader("🚗 Top 10 Mobil Terhemat - Asia")
st.dataframe(top_10_asia[['Make', 'Model', 'Fuel Type Name', 'Comb (L/100 km)']].reset_index(drop=True))

st.subheader("🚗 Top 10 Mobil Terhemat - Eropa")
st.dataframe(top_10_eropa[['Make', 'Model', 'Fuel Type Name', 'Comb (L/100 km)']].reset_index(drop=True))

# Mobil paling hemat
most_asia = top_10_asia.sort_values(by='Comb (L/100 km)').head(1).iloc[0]
most_eropa = top_10_eropa.sort_values(by='Comb (L/100 km)').head(1).iloc[0]

# Tampilkan mobil paling hemat
st.write("**Mobil Terhemat dari Asia:**")
st.success(f"{most_asia['Make']} {most_asia['Model']} ({most_asia['Fuel Type Name']}) - {most_asia['Comb (L/100 km)']} L/100 km")

st.write("**Mobil Terhemat dari Eropa:**")
st.success(f"{most_eropa['Make']} {most_eropa['Model']} ({most_eropa['Fuel Type Name']}) - {most_eropa['Comb (L/100 km)']} L/100 km")

# Bandingkan siapa paling hemat
st.markdown("**🔎 Mobil Paling Hemat di Antara Keduanya:**")
if most_asia['Comb (L/100 km)'] < most_eropa['Comb (L/100 km)']:
    st.info(f"🏆 **{most_asia['Make']} {most_asia['Model']}** dari Asia lebih hemat.")
elif most_asia['Comb (L/100 km)'] > most_eropa['Comb (L/100 km)']:
    st.info(f"🏆 **{most_eropa['Make']} {most_eropa['Model']}** dari Eropa lebih hemat.")
else:
    st.info("Keduanya memiliki efisiensi bahan bakar yang sama.")
