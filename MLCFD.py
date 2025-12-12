# ===============================================
# Machine Learning-Driven CFD Analytics for Wind Energy Assessment
# Coastal Bangladesh Wind Data (2015–2024)
# Author: Md Abid Hassan Mitul
# ===============================================

# --- IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes

# ===============================================
# STEP 1: FETCH REAL-TIME NASA POWER DATA
# ===============================================

# Coastal Bangladesh site coordinates
sites = {
    "CoxBazar": (21.45, 91.97),
    "Chittagong": (22.36, 91.80),
    "Bhola": (22.68, 90.64),
    "Kuakata": (21.82, 90.13),
    "Sundarbans": (21.85, 89.25),
    "Sitakunda": (22.63, 91.67),
    "Noakhali": (22.83, 91.10),
    "Patuakhali": (22.35, 90.33),
    "Mongla": (22.48, 89.60),
    "Khulna": (22.82, 89.55)
}

# NASA POWER API endpoint
base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Function to fetch data for each site
def fetch_nasa_data(lat, lon, start=2015, end=2024):
    params = {
        "start": f"{start}0101",
        "end": f"{end}1231",
        "latitude": lat,
        "longitude": lon,
        "parameters": "WS10M,T2M,PS",
        "format": "CSV",
        "community": "RE"
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"❌ Failed to fetch for lat={lat}, lon={lon}")
        return None
    data = pd.read_csv(io.StringIO(response.text), skiprows=13)
    return data

# Fetch data for all sites
combined_data = []
for site, (lat, lon) in sites.items():
    print(f"Fetching data for {site} ...")
    df = fetch_nasa_data(lat, lon)
    if df is not None:
        df['Site'] = site
        combined_data.append(df)

# Combine all datasets
combined_data = pd.concat(combined_data, ignore_index=True)

# ===============================================
# STEP 2: PREPROCESSING AND DERIVED VARIABLES
# ===============================================
combined_data.columns = ['YEAR', 'MO', 'DY', 'WS10M', 'T2M', 'PS', 'Site']

# Create date column
combined_data['Date'] = pd.to_datetime(dict(
    year=combined_data['YEAR'], month=combined_data['MO'], day=combined_data['DY']
))

# Compute air density (ρ = P / RT) and Wind Power Density (WPD = 0.5 * ρ * v³)
combined_data['rho'] = combined_data['PS'] / (287 * (combined_data['T2M'] + 273.15))
combined_data['WPD'] = 0.5 * combined_data['rho'] * (combined_data['WS10M']**3)

# Month column for monthly analysis
combined_data['Month'] = combined_data['Date'].dt.month

print("✅ Data fetched and processed successfully!")
print("Data shape:", combined_data.shape)
print("Sites:", list(combined_data['Site'].unique()))

# ===============================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ===============================================

sns.set(style="whitegrid", font_scale=1.1)

# --- Figure 1: Average WPD across sites ---
avg_wpd = combined_data.groupby('Site')['WPD'].mean().reset_index().sort_values(by='WPD', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(data=avg_wpd, x='Site', y='WPD', palette='Blues_d')
plt.title("Average Wind Power Density (WPD) Across Coastal Sites (2015–2024)", fontsize=12, fontweight='bold')
plt.ylabel("Average WPD (W/m²)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Figure 2: Wind speed distribution ---
plt.figure(figsize=(10,5))
sns.boxplot(data=combined_data, x='Site', y='WS10M', palette='Blues')
plt.title("Wind Speed Distribution (WS10M) Across Coastal Sites (2015–2024)", fontsize=12, fontweight='bold')
plt.ylabel("Wind Speed at 10 m (m/s)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Figure 3: Monthly mean WPD heatmap ---
monthly_wpd = combined_data.groupby(['Site','Month'])['WPD'].mean().unstack()
plt.figure(figsize=(8,5))
sns.heatmap(monthly_wpd, cmap='YlGnBu', annot=True, fmt=".1f")
plt.title("Monthly Wind Power Density (WPD) Across Coastal Sites (2015–2024)", fontsize=12, fontweight='bold')
plt.xlabel("Month")
plt.ylabel("Site")
plt.tight_layout()
plt.show()

# --- Figure 4: Correlation matrix ---
corr = combined_data[['WS10M', 'T2M', 'PS', 'WPD']].corr()
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix of Meteorological Variables", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Figure 5: Monthly mean wind speed trend ---
monthly_mean = combined_data.groupby(['Date','Site'])['WS10M'].mean().reset_index()
plt.figure(figsize=(9,5))
sns.lineplot(data=monthly_mean, x='Date', y='WS10M', hue='Site')
plt.title("Monthly Mean Wind Speed Trend Across Coastal Sites (2015–2024)", fontsize=12, fontweight='bold')
plt.ylabel("Mean Wind Speed (m/s)")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

# --- Figure 6: Wind Roses (Cox’s Bazar & Bhola) ---
def wind_rose_plot(site_name):
    site_data = combined_data[combined_data['Site'] == site_name]
    ax = WindroseAxes.from_ax()
    ax.bar(site_data['WS10M'], site_data['WPD'], normed=True, opening=0.8, edgecolor='white')
    ax.set_title(f"Wind Rose – {site_name} (2015–2024)", fontsize=10)
    plt.show()

wind_rose_plot('CoxBazar')
wind_rose_plot('Bhola')

# --- Figure 7: Probability distribution of WPD ---
plt.figure(figsize=(8,5))
for site in combined_data['Site'].unique():
    sns.kdeplot(data=combined_data[combined_data['Site'] == site]['WPD'], label=site)
plt.xscale('log')
plt.title("Probability Distribution of Wind Power Density (WPD) Across Coastal Sites (2015–2024)", fontsize=12, fontweight='bold')
plt.xlabel("Wind Power Density (W/m², log scale)")
plt.ylabel("Probability Density")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Figure 8: Seasonal variation in WPD ---
def season_of_month(m):
    if m in [12,1,2]: return 'Winter'
    elif m in [3,4,5]: return 'Pre-Monsoon'
    elif m in [6,7,8]: return 'Monsoon'
    else: return 'Post-Monsoon'

combined_data['Season'] = combined_data['Month'].apply(season_of_month)

plt.figure(figsize=(10,5))
sns.boxplot(data=combined_data, x='Season', y='WPD', hue='Site', palette='viridis')
plt.yscale('log')
plt.title("Seasonal Variation of Wind Power Density (WPD) Across Coastal Sites (2015–2024)", fontsize=12, fontweight='bold')
plt.ylabel("WPD (W/m², log scale)")
plt.tight_layout()
plt.show()

print("✅ EDA completed successfully!")
