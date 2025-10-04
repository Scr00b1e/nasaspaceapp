from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
from pathlib import Path
import gdown  # pip install gdown for Google Drive
import tempfile
import os
import gc  # For garbage collection to close file handles

app = Flask(__name__)
CORS(app)  # Fix CORS for frontend fetches


STATS_FILE_ID = '13_U3-rUSVFbOHv06znZyFds8UbeI8eYA'
EXP_FILE_ID = '1tUEZDWbgqlnb8V1yh3aJS5pvhUcCQtYA'
MODEL_PATH = Path(__file__).parent / 'model.pkl'

# Temporary files for download
temp_stats = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
temp_exp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')

try:
    # Download STATS CSV from Google Drive using gdown (handles virus scan)
    gdown.download(f'https://drive.google.com/uc?id={STATS_FILE_ID}', temp_stats.name, quiet=False)
    df_stats = pd.read_csv(temp_stats.name)
    print(f"STATS loaded: {len(df_stats)} rows, columns: {df_stats.columns.tolist()}")
except Exception as e:
    print(f"STATS download error: {e}")
    df_stats = pd.DataFrame()

try:
    # Download EXP CSV from Google Drive
    gdown.download(f'https://drive.google.com/uc?id={EXP_FILE_ID}', temp_exp.name, quiet=False)
    df_exp = pd.read_csv(temp_exp.name)
    print(f"EXP loaded: {len(df_exp)} rows, columns: {df_exp.columns.tolist()}")
except Exception as e:
    print(f"EXP download error: {e}")
    df_exp = pd.DataFrame()

# Clean up temp files (force close handles with gc)
gc.collect()  # Garbage collect to release file handles on Windows
try:
    os.unlink(temp_stats.name)
    os.unlink(temp_exp.name)
    print("Temp files cleaned up")
except Exception as e:
    print(f"Temp cleanup error: {e} (ignore if files deleted)")

# Global sparse filter: Sample ~10 from Russia, ~65 from rest for worldwide diversity (total ~75 markers)
df = pd.DataFrame()  # Initialize df here
if not df_stats.empty:
    # Force include Russia if available (sample 10)
    russia_mask = df_stats['CTR_MN_NM'].str.contains('Russia', case=False, na=False) if 'CTR_MN_NM' in df_stats.columns else pd.Series([False] * len(df_stats))
    russia_sample = df_stats[russia_mask].sample(min(10, russia_mask.sum()), random_state=42) if russia_mask.sum() > 0 else pd.DataFrame()
    print(f"App.py: Russia rows included: {len(russia_sample)}")
    
    # Rest: Sample from non-Russia for diversity
    non_russia = df_stats[~russia_mask]
    if 'region' in non_russia.columns:
        # Sample 10 per region (excluding Russia)
        sampled_groups = []
        for region, group in non_russia.groupby('region'):
            sample = group.sample(min(10, len(group)), random_state=42)
            sampled_groups.append(sample)
        non_russia_sample = pd.concat(sampled_groups).drop_duplicates()
    else:
        # Fallback: Random 65 from non-Russia
        non_russia_sample = non_russia.sample(min(65, len(non_russia)), random_state=42)
    
    df = pd.concat([russia_sample, non_russia_sample]).drop_duplicates().head(75)  # Cap at 75 total
    print(f"App.py: Filtered to {len(df)} global rows (Russia prioritized, sparse worldwide)")

# Join with EXP for 'P' (pop)
if not df.empty and not df_exp.empty:
    df = pd.merge(df, df_exp[['ID_HDC_G0', 'year', 'P']], on=['ID_HDC_G0', 'year'], how='left')
    df['P'] = df['P'].fillna(1000000)  # Global fallback
else:
    df['P'] = 1000000  # Fixed fallback if no EXP

print(f"Final df columns: {df.columns.tolist()}")  # Debug after merge

# Check columns, set heat_risk = avg_temp * duration / 100 (with fallback)
required_cols = ['avg_temp', 'duration', 'GCPNT_LAT', 'GCPNT_LON']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}, available: {df.columns.tolist()}")  # Debug
    # Fallback mock data if columns missing (for demo)
    df['avg_temp'] = 30.0  # Mock avg temp
    df['duration'] = 2.0  # Mock duration
    df['GCPNT_LAT'] = df.get('GCPNT_LAT', 0)  # Use existing or mock
    df['GCPNT_LON'] = df.get('GCPNT_LON', 0)
    print("Using mock columns for heat_risk calculation")

df['heat_risk'] = df['avg_temp'] * df['duration'] / 100

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Model error: {e}—fallback scaling.")
    model = None

# Format for frontend
def format_data(input_df):
    return [
        {
            "lat": row["GCPNT_LAT"],
            "lon": row["GCPNT_LON"],
            "heat_risk": row["heat_risk"],
            "population": int(row.get("P", 0))  # Ensure int
        }
        for _, row in input_df.iterrows()
    ]

@app.route("/api/heat-data")
def heat_data():
    if df.empty:
        return jsonify([])
    return jsonify(format_data(df))

@app.route("/sim")
def simulate():
    green_increase = float(request.args.get("green", 0))
    simulated_df = df.copy()

    if model:
        current_green = 20.0  # Avg urban green from GEE
        current_df = pd.DataFrame({'green_pct': [current_green]})
        current_heat = model.predict(current_df)[0]
        new_df = pd.DataFrame({'green_pct': [current_green + green_increase]})
        new_heat = model.predict(new_df)[0]
        scale = new_heat / current_heat if current_heat != 0 else 1
    else:
        scale = 1 - green_increase / 100

    simulated_df['heat_risk'] *= scale
    return jsonify({"simulated_data": format_data(simulated_df)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
