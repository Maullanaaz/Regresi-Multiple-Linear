import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load Data ---
try:
    df = pd.read_csv("BAHAN BAKAR MOBIL 2023.csv", encoding="latin1")
except:
    df = pd.read_csv("BAHAN BAKAR MOBIL 2023.csv", encoding="ISO-8859-1")

st.title("📊 Regresi Linear Berganda (Comb (mpg) & CO2 Emissions (g/km) terhadap Fuel Consumption (L/100Km))")

# --- Tampilkan Data Awal ---
st.subheader("📋 Data Awal")
st.dataframe(df)

# --- Variabel X dan Y (Fix sesuai nama kolom) ---
x_vars = ["Comb (mpg)", "CO2 Emissions (g/km)"]
y_var = "Fuel Consumption (L/100Km)"

# --- Tampilkan Data yang Digunakan untuk Regresi ---
st.subheader("📋 Data yang Digunakan untuk Regresi")
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

# --- Visualisasi Prediksi vs Aktual ---
st.subheader("📈 Grafik Prediksi vs Aktual")
fig, ax = plt.subplots()
ax.scatter(y, y_pred, color='green', alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax.set_xlabel("Aktual")
ax.set_ylabel("Prediksi")
ax.set_title(f"Prediksi vs Aktual untuk {y_var}")
st.pyplot(fig)

# --- Insight Tambahan: Mobil Paling Hemat Asia vs Eropa ---
st.markdown("---")
st.subheader("🔍 Insight Tambahan: Perbandingan Mobil Terhemat Asia vs Eropa")

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

# Top 10 mobil paling hemat per region & fuel type
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
