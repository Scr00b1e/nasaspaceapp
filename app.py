from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
from pathlib import Path
import random  # For random sampling

app = Flask(__name__)
CORS(app)  # Fix CORS for frontend fetches

# Relative paths
SCRIPT_DIR = Path(__file__).parent
STATS_PATH = SCRIPT_DIR.parent / 'data' / 'uhe' / 'wbgtmax30-tabular' / 'wbgtmax30_STATS.csv'
EXP_PATH = SCRIPT_DIR.parent / 'data' / 'uhe' / 'wbgtmax30-tabular' / 'wbgtmax30_EXP.csv'  # For population
MODEL_PATH = SCRIPT_DIR / 'model.pkl'

# Load STATS CSV (events)
try:
    df_stats = pd.read_csv(STATS_PATH)
except Exception as e:
    app.logger.error(f"STATS CSV error: {e}")
    df_stats = pd.DataFrame()

# Load EXP CSV (population/exposure)
try:
    df_exp = pd.read_csv(EXP_PATH)
except Exception as e:
    app.logger.error(f"EXP CSV error: {e}")
    df_exp = pd.DataFrame()

# Global sparse filter: Sample ~10 from Russia, ~65 from rest for worldwide diversity (total ~75 markers)
if not df_stats.empty:
    # Force include Russia if available (sample 10)
    russia_mask = df_stats['CTR_MN_NM'].str.contains('Russia', case=False, na=False)
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
    
    df_stats = pd.concat([russia_sample, non_russia_sample]).drop_duplicates().head(75)  # Cap at 75 total
    print(f"App.py: Filtered to {len(df_stats)} global rows (Russia prioritized, sparse worldwide)")

# Join with EXP for 'P' (pop)
if not df_stats.empty and not df_exp.empty:
    df = pd.merge(df_stats, df_exp[['ID_HDC_G0', 'year', 'P']], on=['ID_HDC_G0', 'year'], how='left')
    df['P'] = df['P'].fillna(1000000)  # Global fallback (avg urban pop)
else:
    df = df_stats
    if not df.empty:
        df['P'] = 1000000

# Check columns, set heat_risk = avg_temp * duration / 100
required_cols = ['avg_temp', 'duration', 'GCPNT_LAT', 'GCPNT_LON']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing: {missing}")
df['heat_risk'] = df['avg_temp'] * df['duration'] / 100

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    app.logger.error(f"Model error: {e}—fallback scaling.")
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