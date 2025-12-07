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

# =========================
# Styling (Fraud-app style)
# =========================
st.markdown("""
    <style>
        /* ---------- GLOBAL ---------- */
        .stApp {
            background: radial-gradient(circle at top, #e5f0ff 0, #f1f5ff 40%, #e5f0ff 100%) !important;
            color: #111827 !important;
        }

        html, body, [class*="css"] {
            color: #111827 !important;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Hide sidebar */
        section[data-testid="stSidebar"], div[data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* Top bar */
        header[data-testid="stHeader"] {
            background-color: #e5f0ff00 !important;
        }
        header[data-testid="stHeader"] div {
            box-shadow: none !important;
        }

        /* Main page width */
        div.block-container {
            max-width: 980px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            margin: auto;
        }

        h1 {
            margin-bottom: 0.25rem !important;
        }
        h2 {
            margin-top: 0.75rem !important;
            margin-bottom: 0.4rem !important;
        }

        /* ---------- MAIN CARD ---------- */
        .main-card {
            background-color: #ffffff;
            border-radius: 18px;
            padding: 1.8rem 2.2rem 2.2rem 2.2rem;
            box-shadow: 0 22px 45px rgba(15, 23, 42, 0.18);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        /* Thin divider */
        .soft-divider {
            height: 1px;
            background: linear-gradient(to right, #e5e7eb, #cbd5f5, #e5e7eb);
            margin: 0.8rem 0 1.1rem 0;
        }

        /* ---------- INPUTS & DROPDOWNS ---------- */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div {
            background-color: #ffffff !important;
            color: #111827 !important;
            border-radius: 10px !important;
            border: 1px solid #d3d7e3 !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
            min-height: 42px !important;
        }

        div[data-baseweb="input"]:focus-within > div,
        div[data-baseweb="select"]:focus-within > div,
        div[data-baseweb="textarea"]:focus-within > div {
            border-color: #94b4ff !important;
            box-shadow: 0 0 0 1px #94b4ff55 !important;
        }

        input, textarea {
            color: #111827 !important;
            background-color: #ffffff !important;
        }

        div[data-baseweb="select"] span {
            color: #111827 !important;
        }

        ul[role="listbox"] {
            background-color: #ffffff !important;
            color: #111827 !important;
            border-radius: 10px !important;
            border: 1px solid #d1d5db !important;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
        }

        ul[role="listbox"] li {
            color: #111827 !important;
        }

        svg {
            fill: #6b7280 !important;
        }
            
            /* Make ALL form labels black */
label, 
.stTextInput label, 
.stNumberInput label, 
.stSelectbox label,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stMarkdown"] p {
    color: #111827 !important;
    font-weight: 600 !important;
}


        /* ---- FIXED STREAMLIT PLUS / MINUS BUTTON STYLING ---- */

/* The whole right-side button wrapper */
div[data-baseweb="input"] button {
    background-color: #ffffff !important;  /* white background */
    border-radius: 0px 10px 10px 0px !important;
    border: 1px solid #111827 !important; /* black border */
    width: 42px !important;
    height: 42px !important;
}

/* The + and – icons */
div[data-baseweb="input"] button svg {
    fill: #111827 !important;  /* black icon */
}

/* Hover color */
div[data-baseweb="input"] button:hover {
    background-color: #e5e7eb !important;  /* light gray hover */
}



        /* ---------- PREDICT BUTTON (light grey like screenshot) ---------- */
        .stButton > button {
            background: linear-gradient(180deg, #e5edf7, #d4deee);
            color: #111827 !important;
            border-radius: 9999px !important;
            border: 1px solid #c0c9dd !important;
            padding: 0.55rem 1.6rem !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.18) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(180deg, #dfe7f5, #c8d3ec);
            transform: translateY(-1px);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.22) !important;
        }

        .stButton > button:active {
            transform: translateY(0px);
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.18) !important;
        }

        /* Center the predict button column group a bit tighter */
        .stButton {
            margin-top: 0.4rem;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# Main Card Wrapper
# =========================
# st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("🚗 Car Resale Price Predictor")
st.markdown(
    "Enter the car details below and click **Predict Price** to estimate its resale value."
)

# st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

st.header("🔧 Enter Car Details")

# =========================
# Input Layout (Center, 2 Columns)
# =========================
current_year = datetime.datetime.now().year

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input(
        "Present Price In Showrooms (in lakhs)",
        min_value=0.0,
        max_value=200.0,
        step=0.1,
        value=5.0
    )

    kms_driven = st.number_input(
        "Kms Driven",
        min_value=0,
        max_value=500000,
        step=500,
        value=30000
    )

    owner = st.number_input(
        "Number of previous owners",
        min_value=0,
        max_value=5,
        step=1,
        value=1
    )

with col2:
    car_year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=current_year,
        step=1,
        value=current_year - 5
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )

    seller_type = st.selectbox(
        "Seller Type",
        ["Dealer", "Individual"]
    )

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# Compute no_year (age) internally – model is trained on No_Year
no_year = current_year - car_year

# Centered Predict Button
st.write("")
button_cols = st.columns([1, 1, 1])
with button_cols[1]:
    predict_clicked = st.button("🔍 Predict Price", use_container_width=True)

# =========================
# Prediction Logic
# =========================
if predict_clicked:

    # Manual encoding (must match training)
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

    # Feature order MUST match the order you used during training
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

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.subheader("🔮 Predicted Selling Price")
    st.metric("Estimated Price (lakhs)", f"{prediction:.2f}")

    # =========================
    # Price Comparison
    # =========================
    st.write("")
    st.subheader("📊 Price Comparison")

    comp_df = pd.DataFrame({
        "Category": ["Present Price", "Estimated Selling Price"],
        "Price (lakhs)": [present_price, prediction]
    })

    fig, ax = plt.subplots(figsize=(4, 2))
    sns.barplot(
        x="Category",
        y="Price (lakhs)",
        data=comp_df,
        palette=["#93c5fd", "#fbbf77"],
        ax=ax
    )
    ax.set_ylabel("Price (lakhs)")
    ax.set_xlabel("")
    ax.set_title("Present vs Estimated Price", fontsize=10)
    sns.despine()
    st.pyplot(fig)

    # =========================
    # Model Diagnostics
    # =========================
    st.write("---")
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("### 📉 Actual vs Predicted")

        fig1, ax1 = plt.subplots(figsize=(4, 3))
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

    with dcol2:
        st.markdown("### 📊 Residuals Distribution")

        residuals = y_all - y_pred_all
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.histplot(residuals, bins=30, kde=True, ax=ax2, color="purple")
        ax2.set_xlabel("Prediction Error (lakhs)")
        ax2.set_ylabel("Count")
        ax2.set_title("Residuals Distribution")
        st.pyplot(fig2)

    # =========================
    # Interpretation & Tips
    # =========================
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

# close main-card
# st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.write("---")
st.markdown("Built with ❤️ using **Python, Streamlit, and Scikit-learn**  \nProject by: Ajani")
