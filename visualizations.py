import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ustawienie stylu wykresów
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

df_groundtruth = pd.read_csv("groundtruth.csv", parse_dates=["timestamp"])
df_weather = pd.read_csv("weather_features.csv", parse_dates=["timestamp"])
df_weather = df_weather[(df_weather.loc[:, "tempC"] > -20) & (df_weather.loc[:, "tempC"] < 50)]
df_groundtruth = df_groundtruth[
    (df_groundtruth["occupied"] >= 0)
    & (df_groundtruth["available"] >= 0)
    & (df_groundtruth["occupied"] <= df_groundtruth["max_capacity"])
    & (df_groundtruth["available"] <= df_groundtruth["max_capacity"])
]
df_road = pd.read_csv("road_features.csv")


# =====================================================================
# 2. PRZETWARZANIE I ŁĄCZENIE DANYCH (ETL & PREPROCESSING)
# =====================================================================

# Łączenie tabel (Zgodnie z kluczami z dokumentacji VWFS)
df_joint = pd.merge(df_groundtruth, df_road, on="road_segment_id", how="left")
df_joint = pd.merge(df_joint, df_weather, on=["road_segment_id", "timestamp"], how="left")

# Inżynieria Cech (Feature Engineering)
df_joint["occupancy_rate"] = df_joint["occupied"] / df_joint["max_capacity"]
df_joint["hour"] = df_joint["timestamp"].dt.hour
df_joint["day_of_week"] = df_joint["timestamp"].dt.day_name()
df_joint["day_num"] = df_joint["timestamp"].dt.dayofweek  # Do poprawnego sortowania dni

# =====================================================================
# 3. GENEROWANIE WIZUALIZACJI DO SEKCJI B.1.a
# =====================================================================
"""
# --- WIZUALIZACJA 1: Mapa Ciepła Zajętości w Czasie (Temporal Heatmap) ---
plt.figure(figsize=(14, 7))
heatmap_data = df_joint.sort_values('day_num').pivot_table(
    values='occupancy_rate',
    index='day_of_week',
    columns='hour',
    aggfunc='mean'
)

sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Average Occupancy Rate'})
plt.title('Parking Occupancy Heatmap by Hour and Day of Week', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Day of Week', fontsize=12)
plt.tight_layout()
plt.show()
"""
# --- WIZUALIZACJA 2: Wpływ Warunków Atmosferycznych (Weather Impact) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Wykres A: Temperatura vs Wskaźnik Zajętości
sns.regplot(
    data=df_joint,
    x="tempC",
    y="occupancy_rate",
    ax=axes[0],
    scatter_kws={"alpha": 0.4, "color": "#3498db"},
    line_kws={"color": "red"},
)
axes[0].set_title("Impact of Temperature on Parking Occupancy", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Occupancy Rate")

# Wykres B: Wpływ Opadów Atmosferycznych (Binned for better visibility)
df_joint["rain_category"] = pd.cut(
    df_joint["precipMM"],
    bins=[-0.1, 0.0, 2.0, 10.0],
    labels=["No Rain", "Light Rain", "Heavy Rain"],
)
sns.boxplot(data=df_joint, x="rain_category", y="occupancy_rate", ax=axes[1], palette="Blues")
axes[1].set_title("Impact of Precipitation on Parking Occupancy", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Precipitation Intensity")
axes[1].set_ylabel("Occupancy Rate")

plt.tight_layout()
plt.show()


# --- WIZUALIZACJA 3: Macierz Korelacji Infrastruktury (POI Correlation Matrix) ---
plt.figure(figsize=(10, 8))
poi_columns = [
    "commercial",
    "residential",
    "transportation",
    "schools",
    "eventsites",
    "restaurant",
    "shopping",
    "office",
    "supermarket",
    "num_off_street_parking",
    "off_street_capa",
    "occupancy_rate",
]

# Obliczenie korelacji r-Pearsona dla cech przestrzennych i zmiennej celu
corr_matrix = df_joint[poi_columns].corr()

# Maskowanie górnego trójkąta macierzy dla czytelności
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.5
)
plt.title(
    "Correlation Matrix: Infrastructure Features (POIs) vs. Occupancy Rate",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
