import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================
# Load trained model
# =========================
MODEL_PATH = "random_forest_regression_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# Load dataset for diagnostics
# =========================
# This CSV should have the SAME columns you used for training.
# Example columns:
# ['Selling_Price','Present_Price','Kms_Driven','Owner','No_Year',
#  'Fuel_Type_Diesel','Fuel_Type_Petrol','Seller_Type_Individual','Transmission_Manual']
# Load raw dataset
df = pd.read_csv("car data.csv")

# Drop Car_Name if present
if "Car_Name" in df.columns:
    df = df.drop(columns=["Car_Name"])

# Create No_Year
if "Year" in df.columns:
    df["No_Year"] = datetime.datetime.now().year - df["Year"]
    df = df.drop(columns=["Year"])

# Fuel Type Encoding
df["Fuel_Type_Petrol"] = df["Fuel_Type"].apply(lambda x: 1 if x == "Petrol" else 0)
df["Fuel_Type_Diesel"] = df["Fuel_Type"].apply(lambda x: 1 if x == "Diesel" else 0)
df = df.drop(columns=["Fuel_Type"])

# Seller Type Encoding
df["Seller_Type_Individual"] = df["Seller_Type"].apply(lambda x: 1 if x == "Individual" else 0)
df = df.drop(columns=["Seller_Type"])

# Transmission Encoding
df["Transmission_Manual"] = df["Transmission"].apply(lambda x: 1 if x == "Manual" else 0)
df = df.drop(columns=["Transmission"])

# Ensure correct column order
df = df[[
    "Present_Price",
    "Kms_Driven",
    "Owner",
    "No_Year",
    "Fuel_Type_Diesel",
    "Fuel_Type_Petrol",
    "Seller_Type_Individual",
    "Transmission_Manual",
    "Selling_Price"
]]

# # Separate features and target
# X_all = df.drop(columns=["Selling_Price"])
# y_all = df["Selling_Price"]


# Target and features
y_all = df["Selling_Price"]
X_all = df.drop(columns=["Selling_Price"])

# Model predictions on whole dataset
y_pred_all = model.predict(X_all)

# Metrics
r2 = r2_score(y_all, y_pred_all)
rmse = np.sqrt(mean_squared_error(y_all, y_pred_all))
mae = mean_absolute_error(y_all, y_pred_all)

# =========================
# App Layout
# =========================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Resale Price Predictor")
st.markdown(
    "Predict the **estimated selling price (in lakhs)** of a car based on its specifications "
    "using a trained Random Forest Regression model."
)

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🔧 Enter Car Details")

present_price = st.sidebar.number_input(
    "Present Price In Showrooms (in lakhs)",
    min_value=0.0,
    max_value=200.0,
    step=0.1,
    value=5.0
)

kms_driven = st.sidebar.number_input(
    "Kms Driven",
    min_value=0,
    max_value=500000,
    step=500,
    value=30000
)

owner = st.sidebar.number_input(
    "Number of previous owners",
    min_value=0,
    max_value=5,
    step=1,
    value=1
)

current_year = datetime.datetime.now().year
car_year = st.sidebar.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=current_year,
    step=1,
    value=current_year - 5
)

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

# Compute no_year (age) internally – model is trained on No_Year
no_year = current_year - car_year

# =========================
# Predict button
# =========================
if st.sidebar.button("Predict Price"):

    # -------------------------
    # Manual encoding (must match training)
    # -------------------------
    if fuel_type == "Petrol":
        Fuel_Type_Petrol = 1
        Fuel_Type_Diesel = 0
    elif fuel_type == "Diesel":
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 1
    else:  # CNG or others
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 0

    Seller_Type_Individual = 1 if seller_type == "Individual" else 0
    Transmission_Manual = 1 if transmission == "Manual" else 0

    # Feature order MUST match the order you used during training:
    # ['Present_Price', 'Kms_Driven', 'Owner', 'No_Year',
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

    # Make prediction
    prediction = float(model.predict(input_data)[0])

    # =========================
    # Result metric
    # =========================
    st.subheader("🔮 Predicted Selling Price")
    st.metric("Estimated Price (lakhs)", f"{prediction:.2f}")

    # =========================
    # Price Comparison (smaller chart)
    # =========================
    st.write("---")
    st.subheader("📊 Price Comparison")

    comp_df = pd.DataFrame({
        "Category": ["Present Price", "Estimated Selling Price"],
        "Price (lakhs)": [present_price, prediction]
    })

    fig, ax = plt.subplots(figsize=(3, 1))  # reduced size
    sns.barplot(
        x="Category",
        y="Price (lakhs)",
        data=comp_df,
        palette=["#6AB7A8", "#E9967A"],
        ax=ax
    )
    ax.set_ylabel("Price (lakhs)", fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title("Present vs Estimated Price", fontsize=8)
    sns.despine()
    st.pyplot(fig)

    # =========================
    # Model Diagnostics (Actual vs Predicted + Residuals)
    # =========================
    st.write("---")
    # st.subheader("📈 Model Diagnostics")

    col1, col2 = st.columns(2)

    # 1. Actual vs Predicted
    with col1:
        st.markdown("### 📉 Actual vs Predicted")

        fig1, ax1 = plt.subplots(figsize=(3, 3))
        sns.scatterplot(x=y_all, y=y_pred_all, alpha=0.6, ax=ax1)
        ax1.plot(
            [y_all.min(), y_all.max()],
            [y_all.min(), y_all.max()],
            "r--",
            linewidth=1.5
        )
        ax1.set_xlabel("Actual Selling Price (lakhs)")
        ax1.set_ylabel("Predicted Selling Price (lakhs)")
        ax1.set_title(f"R² = {r2:.3f}")
        st.pyplot(fig1)

    # 2. Residuals Distribution
    with col2:
        st.markdown("### 📊 Residuals Distribution")

        residuals = y_all - y_pred_all
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        sns.histplot(residuals, bins=30, kde=True, ax=ax2, color="purple")
        ax2.set_xlabel("Prediction Error (lakhs)")
        ax2.set_ylabel("Count")
        ax2.set_title("Residuals Distribution")
        st.pyplot(fig2)


    st.write("---")
    st.subheader("💡 Interpretation & Tips")

    ratio = prediction / present_price if present_price > 0 else 1

    if 0.9 <= ratio <= 1.1:
        st.markdown(
            f"- The predicted price (**{prediction:.2f} lakhs**) is close to the present price (**{present_price:.2f} lakhs**).  \n"
            "- This suggests your current expectation is **in line with market trends**."
        )
    elif ratio < 0.9:
        st.markdown(
            f"- The predicted price (**{prediction:.2f} lakhs**) is **lower** than the present price (**{present_price:.2f} lakhs**).  \n"
            "- You may need to **reduce your asking price** or highlight more features of the car."
        )
    else:
        st.markdown(
            f"- The predicted price (**{prediction:.2f} lakhs**) is **higher** than the present price (**{present_price:.2f} lakhs**).  \n"
            "- You might be **undervaluing** your car, or it’s in better condition than average similar vehicles."
        )

    st.markdown(
        """
        **General Tips to Improve Resale Value:**
        - Keep complete **service and maintenance records**  
        - Ensure the car is **clean, polished, and dent-free** before selling  
        - Fix small issues (lights, wipers, scratches) before listing  
        - Highlight **low mileage, single owner, and original parts** in your ad  
        - Compare your price with similar listings on car sale platforms  
        """
    )

# =========================
# Footer
# =========================
st.write("---")
st.markdown("Built with ❤️ using **Python, Streamlit, and Scikit-learn**  \nProject by: Ajani")
