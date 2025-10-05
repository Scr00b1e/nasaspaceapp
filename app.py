from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
from pathlib import Path
import gdown  # pip install gdown for Google Drive
import tempfile
import os
import gc  # For garbage collection to close file handles
import random  # For random sampling

app = Flask(__name__)
CORS(app)  # Fix CORS for frontend fetches

# Google Drive file IDs (extract from your URLs)
STATS_FILE_ID = '13_U3-rUSVFbOHv06znZyFds8UbeI8eYA'
EXP_FILE_ID = '1tUEZDWbgqlnb8V1yh3aJS5pvhUcCQtYA'
MODEL_PATH = Path(__file__).parent / 'model.pkl'  # Model still local

# Temporary files for download
temp_stats = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
temp_exp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')

# Download STATS CSV from Google Drive using gdown (handles virus scan)
try:
    gdown.download(f'https://drive.google.com/uc?id={STATS_FILE_ID}', temp_stats.name, quiet=False)
    print(f"STATS downloaded to temp: {os.path.getsize(temp_stats.name) / (1024*1024):.1f} MB")
except Exception as e:
    print(f"STATS download error: {e}")
    df_stats = pd.DataFrame()

# Download EXP CSV from Google Drive
try:
    gdown.download(f'https://drive.google.com/uc?id={EXP_FILE_ID}', temp_exp.name, quiet=False)
    print(f"EXP downloaded to temp: {os.path.getsize(temp_exp.name) / (1024*1024):.1f} MB")
except Exception as e:
    print(f"EXP download error: {e}")
    df_exp = pd.DataFrame()

# Chunked load for STATS (low memory, sample 150 random rows)
df_stats = pd.DataFrame()
if os.path.exists(temp_stats.name):
    try:
        chunks = pd.read_csv(temp_stats.name, chunksize=1000)  # 1k rows per chunk
        all_chunks = []
        for chunk in chunks:
            all_chunks.append(chunk.sample(min(10, len(chunk)), random_state=random.randint(1, 1000)))  # Sample 10 per chunk
            del chunk  # Discard chunk
        df_stats = pd.concat(all_chunks, ignore_index=True).drop_duplicates().head(150)  # Total 150 unique
        print(f"STATS loaded chunked: {len(df_stats)} rows, columns: {df_stats.columns.tolist()}")
    except Exception as e:
        print(f"STATS chunk error: {e}")
        df_stats = pd.DataFrame()
    finally:
        gc.collect()  # GC to free memory

# Chunked load for EXP (sample 150)
df_exp = pd.DataFrame()
if os.path.exists(temp_exp.name):
    try:
        chunks = pd.read_csv(temp_exp.name, chunksize=1000)  # 1k rows per chunk
        all_chunks = []
        for chunk in chunks:
            all_chunks.append(chunk.sample(min(10, len(chunk)), random_state=random.randint(1, 1000)))  # Sample 10 per chunk
            del chunk  # Discard chunk
        df_exp = pd.concat(all_chunks, ignore_index=True).drop_duplicates().head(150)  # Total 150 unique
        print(f"EXP loaded chunked: {len(df_exp)} rows, columns: {df_exp.columns.tolist()}")
    except Exception as e:
        print(f"EXP chunk error: {e}")
        df_exp = pd.DataFrame()
    finally:
        gc.collect()  # GC to free memory

# Clean up temp files
try:
    os.unlink(temp_stats.name)
    os.unlink(temp_exp.name)
    print("Temp files cleaned up")
except Exception as e:
    print(f"Temp cleanup error: {e} (ignore if files deleted)")

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
    if df_stats.empty:
        return jsonify([])
    
    # Use the chunked df_stats (150 rows)
    df_sample = df_stats.copy()
    
    # Join with EXP for 'P' (pop)
    if not df_exp.empty:
        df_sample = pd.merge(df_sample, df_exp[['ID_HDC_G0', 'year', 'P']], on=['ID_HDC_G0', 'year'], how='left')
        df_sample['P'] = df_sample['P'].fillna(1000000)  # Global fallback
    else:
        df_sample['P'] = 1000000  # Fixed fallback

    # Check columns, set heat_risk = avg_temp * duration / 100
    required_cols = ['avg_temp', 'duration', 'GCPNT_LAT', 'GCPNT_LON']
    missing = [col for col in required_cols if col not in df_sample.columns]
    if missing:
        print(f"Missing columns: {missing}, available: {df_sample.columns.tolist()}")  # Debug
        # Fallback mock data if columns missing (for demo)
        df_sample['avg_temp'] = 30.0  # Mock avg temp
        df_sample['duration'] = 2.0  # Mock duration
        df_sample['GCPNT_LAT'] = df_sample.get('GCPNT_LAT', 0)  # Use existing or mock
        df_sample['GCPNT_LON'] = df_sample.get('GCPNT_LON', 0)
        print("Using mock columns for heat_risk calculation")

    df_sample['heat_risk'] = df_sample['avg_temp'] * df_sample['duration'] / 100

    return jsonify(format_data(df_sample))

@app.route("/sim")
def simulate():
    green_increase = float(request.args.get("green", 0))
    # For sim, use the last loaded df_stats (global, or regenerate if needed)
    simulated_df = df_stats.copy()  # From startup chunked sample

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
