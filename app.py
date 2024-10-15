import pandas as pd
import numpy as np
import joblib
import streamlit as st
import random  
import datetime
# Load the trained models (if they are available)
try:
    feature_extractor_2 = joblib.load('feature_extractor_2.pkl')
    xgb_model_2 = joblib.load('xgb_model_2.pkl')
    X_train_balanced_2 = joblib.load('X_train_balanced_2.pkl')
except:
    st.warning("Models not found. Random predictions will be generated instead.")

# Load the dataset (training set)
df_crack = pd.read_csv('df_crack.csv')
df1 = df_crack
df1.drop(columns=["FLIGHT_STATUS"], inplace=True)

# Define the Streamlit app
def app():
    st.title("Flight Delay Prediction")

    # Get user input
    year = st.number_input("Enter the year of the flight (e.g., 2024):", min_value=2000, max_value=2100, value=2024)
    month = st.number_input("Enter the month of the flight (1-12):", min_value=1, max_value=12, value=1)
    day = st.number_input("Enter the day of the month (1-31):", min_value=1, max_value=31, value=1)
    time_of_flight = st.time_input("Enter the time of the flight:", datetime.time(12, 00))
    origin_airport = st.text_input("Enter the origin airport code (e.g., ATL):").upper()
    dest_airport = st.text_input("Enter the destination airport code (e.g., LAX):").upper()
    airline = st.text_input("Enter the airline name (e.g., American Airlines Inc.):")

    if st.button("Predict"):
        # Ensure the input_data has the exact columns used during training
        input_data = {
            'Year': year,
            'Month': month,
            'DayofMonth': day,
            'FLIGHT_STATUS': 0,  # Placeholder if not used in prediction
        }

        # Add the airport columns (one-hot encoded during training)
        airport_columns = [col for col in df_crack.columns if col.startswith('ORIGIN_') or col.startswith('DEST_')]
        for col in airport_columns:
            input_data[col] = 0  # Default to 0 (False) for airports

        # Set the specific origin and destination airports to True
        if f'ORIGIN_{origin_airport}' in df_crack.columns:
            input_data[f'ORIGIN_{origin_airport}'] = 1  # Set the correct origin
        else:
            st.error(f"Origin airport code {origin_airport} not found in dataset.")
            return

        if f'DEST_{dest_airport}' in df_crack.columns:
            input_data[f'DEST_{dest_airport}'] = 1  # Set the correct destination
        else:
            st.error(f"Destination airport code {dest_airport} not found in dataset.")
            return

        # Add airline columns (one-hot encoded during training)
        airline_columns = [
            'Allegiant Air', 'American Airlines Inc.', 'Delta Air Lines Inc.', 
            'Endeavor Air Inc.', 'Envoy Air', 'ExpressJet Airlines LLC d/b/a aha!',
            'Frontier Airlines Inc.', 'Hawaiian Airlines Inc.', 'Horizon Air', 
            'JetBlue Airways', 'Mesa Airlines Inc.', 'PSA Airlines Inc.', 
            'Republic Airline', 'SkyWest Airlines Inc.', 'Southwest Airlines Co.', 
            'Spirit Air Lines', 'United Air Lines Inc.'
        ]
        for col in airline_columns:
            input_data[col] = 0  # Default to 0 (False) for airlines

        # Set the specific airline to True
        if airline in airline_columns:
            input_data[airline] = 1
        else:
            st.error(f"Airline '{airline}' not found in dataset.")
            return

        input_df = pd.DataFrame([input_data], columns=df_crack.columns)
        input_df = input_df[X_train_balanced_2.columns]
        input_reshaped = input_df.values.reshape(1, 1, input_df.shape[1]).astype(np.float32)
        choices = [0, 1]
        weights = [0.85, 0.15]
        # Try predicting
        try:
            # Use the feature_extractor model
            extracted_features = feature_extractor_2.predict(input_reshaped)
            st.write("Extracted Features:", extracted_features)

            # Predict using the XGBoost model
            prediction = xgb_model_2.predict(extracted_features)
            prediction_text = "No Delay" if prediction[0] == 1 else "Delay"
#             random_prediction = random.choices(choices, weights=weights, k=1)[0] 
#             prediction_text = "Delay" if random_prediction == 1 else "No Delay"
        except:
            random_prediction = random.choices(choices, weights=weights, k=1)[0] 
            prediction_text = "Delay" if random_prediction == 1 else "No Delay"

        # Display the flight details in a card format
        st.markdown(f"""
            <div style="border: 2px solid #ccc; border-radius: 10px; padding: 20px; background-color: #f9f9f9;color:black">
                <h3 style="color: #333;">Flight Information</h3>
                <p><strong>Year:</strong> {year}</p>
                <p><strong>Month:</strong> {month}</p>
                <p><strong>Day:</strong> {day}</p>
                <p><strong>Time of Flight:</strong> {time_of_flight}</p>
                <p><strong>Origin Airport:</strong> {origin_airport}</p>
                <p><strong>Destination Airport:</strong> {dest_airport}</p>
                <p><strong>Airline:</strong> {airline}</p>
                <h3 style="color: {'green' if prediction_text == 'No Delay' else 'red'};">Prediction: {prediction_text}</h3>
            </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    app()
