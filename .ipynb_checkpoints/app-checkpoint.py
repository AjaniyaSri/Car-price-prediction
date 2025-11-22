import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------
# make sure this file name matches your actual .pkl file
MODEL_PATH = "random_forest_regression_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown("""
    <style>
        div[data-baseweb="select"] span {
            cursor: pointer !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to get an estimated selling price.")

st.sidebar.header("Car Details")

# Numeric inputs
present_price = st.sidebar.number_input(
    "Present Price (in lakhs)",
    min_value=0.0,
    value=5.0,
    step=0.1,
    help="Current ex-showroom price of the car (in lakhs)"
)

kms_driven = st.sidebar.number_input(
    "Kms Driven",
    min_value=0,
    value=30000,
    step=500,
    help="Total kilometers driven"
)

owner = st.sidebar.selectbox(
    "Number of previous owners",
    [0, 1, 2, 3],
    index=0
)

car_age = st.sidebar.number_input(
    "Age of Car (years)",
    min_value=0,
    value=5,
    step=1,
    help="How many years old is the car?"
)

# Categorical inputs
fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.sidebar.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.sidebar.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

st.write("### Entered Details")
st.write(f"- Present Price: **{present_price} lakhs**")
st.write(f"- Kms Driven: **{kms_driven} km**")
st.write(f"- Owners: **{owner}**")
st.write(f"- Car Age: **{car_age} years**")
st.write(f"- Fuel Type: **{fuel_type}**")
st.write(f"- Seller Type: **{seller_type}**")
st.write(f"- Transmission: **{transmission}**")

# -----------------------------
# Manual encoding (must match training)
# -----------------------------
# Assuming you created dummy variables like:
# Fuel_Type_Diesel, Fuel_Type_Petrol
# Seller_Type_Individual
# Transmission_Manual

# Fuel type encoding
if fuel_type == "Petrol":
    Fuel_Type_Petrol = 1
    Fuel_Type_Diesel = 0
elif fuel_type == "Diesel":
    Fuel_Type_Petrol = 0
    Fuel_Type_Diesel = 1
else:  # CNG or others
    Fuel_Type_Petrol = 0
    Fuel_Type_Diesel = 0

# Seller type encoding
Seller_Type_Individual = 1 if seller_type == "Individual" else 0

# Transmission encoding
Transmission_Manual = 1 if transmission == "Manual" else 0

# Age column name is usually "no_year" or similar in tutorials
no_year = car_age

# -----------------------------
# Prediction
# -----------------------------
# Feature order MUST match the order used during training
# Example order:
# ['Present_Price', 'Kms_Driven', 'Owner', 'no_year',
#  'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
#  'Seller_Type_Individual', 'Transmission_Manual']

input_data = np.array([[
    present_price,
    kms_driven,
    owner,
    no_year,
    Fuel_Type_Diesel,
    Fuel_Type_Petrol,
    Seller_Type_Individual,
    Transmission_Manual
]])

if st.button("Predict Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: **{prediction:.2f} lakhs**")
        st.caption("Note: This is an approximate estimate based on the trained model.")
    except Exception as e:
        st.error("❌ Something went wrong while predicting.")
        st.exception(e)
