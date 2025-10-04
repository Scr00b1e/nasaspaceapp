import ee
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# One-time auth (run if not done; opens browser)
ee.Authenticate()
ee.Initialize()

# Step 1: Load SEDAC CSV (from wbgtmax30-tabular; adapt exact filename)
csv_path = 'urbavitality/data/uhe/wbgtmax30-tabular/wbgtmax30_STATS'  # Your file here
df_sedac = pd.read_csv(csv_path)
print("SEDAC columns:", df_sedac.columns.tolist())  # Debug: Check if 'UC_NM_MN', 'tot_days' exist

# Filter for Los Angeles (handles variants like "Los Angeles--Long Beach--Anaheim")
la_data = df_sedac[df_sedac['UC_NM_MN'].str.contains('Los Angeles', case=False, na=False)]  # Swap 'UC_NM_MN' if col differs
if la_data.empty:
    print("No LA data? Sample city names:", df_sedac['UC_NM_MN'].unique()[:10])
    exit()

# Extract years and heat_score (tot_days: extreme days/year)
la_data = la_data[(la_data['year'] >= 1983) & (la_data['year'] <= 2016)]  # Ensure range
years = sorted(la_data['year'].unique())
heat_scores = la_data.set_index('year')['tot_days'].reindex(years, fill_value=0).values  # Align to years
print(f"LA SEDAC: {len(years)} years, avg heat_score (extreme days/yr): {heat_scores.mean():.2f}")

# Step 2: GEE - Query yearly avg NDVI (green %) for LA over 1983-2016
la_bounds = ee.Geometry.Rectangle([-118.5, 33.7, -117.7, 34.3])  # LA metro box

# 10 sample points: LA neighborhoods (lon, lat; from maps)
point_coords = [
    [-118.24, 34.05],  # Downtown
    [-118.25, 34.06],  # Echo Park
    [-118.41, 33.99],  # South LA
    [-118.33, 34.09],  # Koreatown
    [-118.29, 34.10],  # Hollywood
    [-118.44, 34.05],  # Inglewood
    [-118.15, 34.05],  # Pasadena
    [-118.35, 34.00],  # Compton
    [-118.20, 34.00],  # Long Beach
    [-118.30, 34.15]   # Burbank
]
points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(coord), {'id': i}) for i, coord in enumerate(point_coords)])

def get_yearly_ndvi(year):
    start = f'{year}-01-01'
    end = f'{year}-12-31'
    coll = (ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
            .filterBounds(la_bounds)
            .filterDate(start, end)
            .filter(ee.Filter.lt('CLOUD_COVER', 50))  # Low cloud
            .map(lambda img: img.normalizedDifference(['B5', 'B4']).rename('ndvi'))
            .mean())  # Avg NDVI for year
    return coll.clip(la_bounds)

# Extract yearly avg NDVI at points (mean across 10 points for city-level)
green_pcts = []
for year in years:
    yearly_img = get_yearly_ndvi(year)
    sampled = yearly_img.reduceRegions(points, ee.Reducer.mean(), 30)
    info = sampled.getInfo()['features']
    if info:
        avg_ndvi = sum(f['properties']['ndvi'] for f in info if f['properties']['ndvi'] is not None) / len([f for f in info if f['properties']['ndvi'] is not None])
        green_pcts.append(avg_ndvi * 100 if avg_ndvi else 15)  # %; fallback
    else:
        green_pcts.append(15)  # Fallback if no data
print(f"LA GEE: Avg green_pct over period: {pd.Series(green_pcts).mean():.2f}%")

# Step 3: Build DF (pair years: green_pct ~ heat_score)
df = pd.DataFrame({
    'year': years,
    'green_pct': green_pcts,
    'heat_score': heat_scores
})
df = df.dropna()  # Clean

# Step 4: Train & save model
if len(df) < 2:
    print("Not enough data for model—using mock.")
    df = pd.DataFrame({'green_pct': [10, 20, 30], 'heat_score': [90, 75, 60]})
X = df[['green_pct']]
y = df['heat_score']
model = LinearRegression().fit(X, y)
joblib.dump(model, 'model.pkl')

print('SEDAC + GEE model saved! Sample DF:')
print(df.head())
print(f'Test predict +20% green (from avg {df["green_pct"].mean():.1f}%): {model.predict([[df["green_pct"].mean() + 20]])[0]:.2f} extreme days/yr')