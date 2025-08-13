import os
os.environ["HADOOP_OPTS"] = "-Djava.library.path="
os.environ["HADOOP_HOME"] = "C:/hadoop"

from pyspark.sql import SparkSession

import streamlit as st
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import math
import os
from dotenv import load_dotenv

import numpy as np
import joblib


# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load zone lookup table
zone_lookup = pd.read_csv("datasets/taxi_zone_lookup.csv")
zone_lookup = zone_lookup.drop_duplicates(subset=["LocationID"])

# Load taxi zones shapefile
taxi_zones = gpd.read_file("taxi_zones/taxi_zones.shp")

# Convert since Google geocoding returns WGS84 (EPSG:4326, lat/lon) and NYC Taxi Zone Shapefile is not lat/lon
taxi_zones = taxi_zones.to_crs(epsg=4326) 

# Load average trip distance lookup (precomputed based on raw data)
avg_dist_lookup = pd.read_csv("datasets/avg_trip_distance_lookup.csv")  # columns: PULocationID, DOLocationID, trip_distance

def assign_cluster(location_id, centers):
    dists = np.abs(centers.flatten() - location_id)
    return int(np.argmin(dists))

pu_centers = np.load('trained_models/pu_kmeans_centers.npy')
do_centers = np.load('trained_models/do_kmeans_centers.npy')

def geocode_address(address, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    resp = requests.get(url).json()
    print(f"Geocoding '{address}': {resp}")  # Debug print
    if resp['status'] == 'OK':
        loc = resp['results'][0]['geometry']['location']
        return loc['lat'], loc['lng']
    return None, None

def latlon_to_zone(lat, lon):
    print(f"Mapping lat/lon to zone: lat={lat}, lon={lon}")  # Debug print
    point = Point(lon, lat)
    match = taxi_zones[taxi_zones.geometry.contains(point)]
    print(f"Zones found: {match}")  # Debug print
    if not match.empty:
        row = match.iloc[0]
        return int(row['LocationID']), row['borough'], row['zone']
    return None, None, None

def get_avg_distance(pu, do):
    row = avg_dist_lookup[(avg_dist_lookup['PULocationID'] == pu) & (avg_dist_lookup['DOLocationID'] == do)]
    if not row.empty:
        return float(row['trip_distance'].values[0])
    else:
        return 1.62  # fallback to median


# --- STREAMLIT UI ---
st.set_page_config(
    page_title="NYC Yellow Taxi Fare Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Sidebar user guide/about section
st.sidebar.markdown("""
**About this Project**

This app predicts NYC Yellow Taxi fares using a machine learning model trained on 2023 data.

- Enter pickup and dropoff addresses, time, and other trip details to get a fare estimate.
- The model uses advanced feature engineering and the top 20 most important features for accurate predictions.
- Built with Streamlit, scikit-learn, and geospatial data.
""")


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Fare", "Feature List", "Performance"])

st.title("NYC Yellow Taxi Fare Prediction App")

if page == "Predict Fare":
    st.markdown("Enter your trip details below:")
    with st.form("trip_form"):
        pickup_address = st.text_input("Pickup Address (or Place Name)", "")
        dropoff_address = st.text_input("Dropoff Address (or Place Name)", "")
        pickup_date = st.date_input("Pickup Date", value=datetime.today())
        # 12-hour format with AM/PM
        pickup_hour_12 = st.selectbox("Pickup Hour", list(range(1, 13)), index=11)
        pickup_ampm = st.selectbox("AM/PM", ["AM", "PM"], index=1)
        # Convert to 24-hour format
        if pickup_ampm == "AM":
            pickup_hour = 0 if pickup_hour_12 == 12 else pickup_hour_12
        else:
            pickup_hour = 12 if pickup_hour_12 == 12 else pickup_hour_12 + 12
        passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1, step=1)
        payment_type = st.selectbox("Payment Type", ["Credit Card", "Cash", "No Charge", "Dispute", "Unknown"])
        ratecode = st.selectbox("Rate Code", [
            "1 - Standard Rate", "2 - JFK", "3 - Newark", "4 - Nassau/Westchester", "5 - Negotiated", "6 - Group Ride"
        ], index=0)
        submit = st.form_submit_button("Predict Fares")

        if submit:
            # ...existing fare prediction logic...
            # Geocode addresses
            pickup_lat, pickup_lon = geocode_address(pickup_address, GOOGLE_API_KEY)
            dropoff_lat, dropoff_lon = geocode_address(dropoff_address, GOOGLE_API_KEY)

            if None in (pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
                st.error("Could not geocode one or both addresses. Please check your input.")
            else:
                # Map to taxi zones using shapefile
                pickup_id, pickup_borough, pickup_zone = latlon_to_zone(pickup_lat, pickup_lon)
                dropoff_id, dropoff_borough, dropoff_zone = latlon_to_zone(dropoff_lat, dropoff_lon)
                ratecode_id = int(ratecode.split(" - ")[0])

                # Get average trip distance for this route
                trip_distance = get_avg_distance(pickup_id, dropoff_id)

                # Assign pickup and dropoff clusters using numpy and exported centers
                pickup_cluster = assign_cluster(pickup_id, pu_centers)
                dropoff_cluster = assign_cluster(dropoff_id, do_centers)
                same_cluster_trip = int(pickup_cluster == dropoff_cluster)
                cluster_diff = abs(pickup_cluster - dropoff_cluster)

                # --- FULL FEATURE ENGINEERING ---
                # Time features
                pickup_datetime = datetime.combine(pickup_date, datetime.min.time()).replace(hour=pickup_hour)
                dow = pickup_datetime.weekday()
                month = pickup_datetime.month
                dow_sin = math.sin(2 * math.pi * dow / 7)
                dow_cos = math.cos(2 * math.pi * dow / 7)
                month_sin = math.sin(2 * math.pi * month / 12)
                month_cos = math.cos(2 * math.pi * month / 12)
                hour_sin = math.sin(2 * math.pi * pickup_hour / 24)
                hour_cos = math.cos(2 * math.pi * pickup_hour / 24)

                # Interaction terms and flags
                distance_per_passenger = trip_distance / passenger_count if passenger_count > 0 else 0.0
                cluster_distance_interaction = trip_distance * cluster_diff
                is_standard_rate = int(ratecode_id == 1)
                is_airport_rate = int(ratecode_id in [2,3])
                is_cash_payment = int(payment_type.lower() == "cash")
                is_rush_hour = int((7 <= pickup_hour <= 9) or (16 <= pickup_hour <= 19))
                rush_hour_distance = is_rush_hour * trip_distance
                is_weekend = int(dow >= 5)
                is_late_night = int(pickup_hour >= 22 or pickup_hour < 6)
                is_premium_route = int((pickup_cluster == 8) or (dropoff_cluster == 8))
                weekend_premium_route = is_weekend * is_premium_route
                cluster_passenger_interaction = pickup_cluster * passenger_count

                # pickup_dropoff_combo features (example: pickup_dropoff_combo_2_7)
                pickup_dropoff_combo = f"pickup_dropoff_combo_{pickup_cluster}_{dropoff_cluster}"
                pickup_dropoff_combo_2_7 = int(pickup_dropoff_combo == "pickup_dropoff_combo_2_7")
                pickup_dropoff_combo_3_0 = int(pickup_dropoff_combo == "pickup_dropoff_combo_3_0")

                # Map payment_type string to numeric code
                payment_type_map = {
                    "Credit Card": 1,
                    "Cash": 2,
                    "No Charge": 3,
                    "Dispute": 4,
                    "Unknown": 5
                }
                payment_type_code = payment_type_map.get(payment_type, 5)

                # Build input DataFrame for prediction
                input_dict = {
                    'trip_distance': trip_distance,
                    'RatecodeID': ratecode_id,
                    'payment_type': payment_type_code,
                    'is_standard_rate': is_standard_rate,
                    'hour_sin': hour_sin,
                    'hour_cos': hour_cos,
                    'DOLocationID': dropoff_id,
                    'pickup_dropoff_combo_2_7': pickup_dropoff_combo_2_7,
                    'distance_per_passenger': distance_per_passenger,
                    'is_airport_rate': is_airport_rate,
                    'PULocationID': pickup_id,
                    'is_cash_payment': is_cash_payment,
                    'rush_hour_distance': rush_hour_distance,
                    'month_cos': month_cos,
                    'month_sin': month_sin,
                    'cluster_distance_interaction': cluster_distance_interaction,
                    'cluster_diff': cluster_diff,
                    'dow_cos': dow_cos,
                    'pickup_dropoff_combo_3_0': pickup_dropoff_combo_3_0,
                    'dow_sin': dow_sin
                }

                # Load the trained model and feature columns
                model = joblib.load('trained_models/sklearn_rf/random_forest_regressor.pkl')
                feature_columns = joblib.load('trained_models/sklearn_rf/feature_columns.pkl')

                # Ensure all features are present and in correct order
                X_input = pd.DataFrame([input_dict])
                missing_cols = [col for col in feature_columns if col not in X_input.columns]
                for col in missing_cols:
                    X_input[col] = 0
                X_input = X_input[feature_columns]

                # Predict and display results
                log_pred = model.predict(X_input)[0]
                st.success(f'Predicted log(fare): {log_pred:.2f}')
                st.info(f'Predicted fare: ${np.exp(log_pred):.2f}')

elif page == "Feature List":
    feature_columns = joblib.load('trained_models/sklearn_rf/feature_columns.pkl')
    import matplotlib.pyplot as plt
    import seaborn as sns
    importances = None
    try:
        model = joblib.load('trained_models/sklearn_rf/random_forest_regressor.pkl')
        importances = model.feature_importances_
    except Exception as e:
        st.warning(f"Could not load model or importances: {e}")

    if importances is not None:
        import pandas as pd
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)  # Most important at top

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
        ax.set_title('Top 20 Feature Importances', fontsize=10, weight='bold')
        ax.set_xlabel('Importance', fontsize=8)
        ax.set_ylabel('Feature', fontsize=8)
        ax.tick_params(axis='both', labelsize=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Feature importances not available.")

elif page == "Performance":
    st.subheader("Model Performance Metrics")
    # Try to load metrics if saved, else show placeholder
    import os
    perf_path = 'trained_models/sklearn_rf/model_performance.json'
    if os.path.exists(perf_path):
        import json
        with open(perf_path, 'r') as f:
            metrics = json.load(f)
        st.write("**RandomForestRegressor (top 20 features):**")
        st.write(f"- RMSE: {metrics.get('rmse', 'N/A')}")
        st.write(f"- MAE: {metrics.get('mae', 'N/A')}")
        st.write(f"- R²: {metrics.get('r2', 'N/A')}")
    else:
        st.info("Model performance metrics not found. To display metrics, save them as a JSON file at 'trained_models/sklearn_rf/model_performance.json' with keys 'rmse', 'mae', and 'r2'. Example: {\"rmse\": 0.25, \"mae\": 0.18, \"r2\": 0.82}")

    # Additional Visuals: Actual vs. Predicted and Residuals Plots
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import joblib
        import numpy as np
        import pandas as pd
        test_data = pd.read_csv("datasets/test_sample_for_streamlit.csv")
        feature_cols = joblib.load("trained_models/sklearn_rf/feature_columns.pkl")
        model = joblib.load("trained_models/sklearn_rf/random_forest_regressor.pkl")
        # Ensure all expected columns are present in test_data
        missing_cols = [col for col in feature_cols if col not in test_data.columns]
        for col in missing_cols:
            test_data[col] = 0
        # Reorder columns to match feature_cols
        X = test_data[feature_cols]
        y_true = test_data["log_total_amount"]
        y_pred = model.predict(X)
        # For original scale
        y_true_exp = np.exp(y_true)
        y_pred_exp = np.exp(y_pred)
        residuals = y_true_exp - y_pred_exp


        # Side-by-side Actual vs. Predicted plots
        st.subheader("Actual vs. Predicted (Log Scale) vs (Original Scale)")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        # Log scale
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, s=10, ax=ax1)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7)
        ax1.set_xlabel("Actual log(Fare)", fontsize=9)
        ax1.set_ylabel("Predicted log(Fare)", fontsize=9)
        ax1.set_title("Log Scale", fontsize=10)
        ax1.tick_params(axis='both', labelsize=8)
        ax1.grid(True, alpha=0.3)
        # Original scale
        sns.scatterplot(x=y_true_exp, y=y_pred_exp, alpha=0.3, s=10, ax=ax2)
        min_val_real = min(y_true_exp.min(), y_pred_exp.min())
        max_val_real = max(y_true_exp.max(), y_pred_exp.max())
        ax2.plot([min_val_real, max_val_real], [min_val_real, max_val_real], 'r--', lw=2, alpha=0.7)
        ax2.set_xlabel("Actual Fare ($)", fontsize=9)
        ax2.set_ylabel("Predicted Fare ($)", fontsize=9)
        ax2.set_title("Original Scale", fontsize=10)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # Residuals Plot (Original Scale)
        st.subheader("Residuals Plot (Original Scale)")
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4))
        # Residuals vs Predicted
        sns.scatterplot(x=y_pred_exp, y=residuals, alpha=0.3, s=10, ax=ax2)
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_xlabel("Predicted Fare ($)", fontsize=9)
        ax2.set_ylabel("Residuals ($)", fontsize=9)
        ax2.set_title("Residuals vs. Predicted (Original Scale)", fontsize=10)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.grid(True, alpha=0.3)
        # Residuals histogram
        sns.histplot(residuals, bins=50, kde=True, ax=ax3, color='skyblue')
        ax3.axvline(0, color='red', linestyle='--')
        ax3.set_xlabel("Residuals ($)", fontsize=9)
        ax3.set_ylabel("Frequency", fontsize=9)
        ax3.set_title("Residuals Distribution", fontsize=10)
        ax3.tick_params(axis='both', labelsize=8)
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig2)

        st.caption("Visuals above: (1) Actual vs. Predicted fares in log scale, (2) Actual vs. Predicted fares in original scale, (3) Residuals vs. predicted and residuals distribution in original scale. Ideally, points should cluster along the diagonal and residuals should be centered around zero.")
    except Exception as e:
        st.warning("Could not display performance plots. Ensure test sample, model, and feature columns are available.")
        st.text(str(e))
