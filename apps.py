import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

st.title("💧 Water Potability Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload Water Potability Dataset", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Clean dataset
    df_clean = df.fillna(df.mean())

    # Features and Target
    X = df_clean.drop(columns=['Potability'])
    y = df_clean['Potability']

    # Balance dataset
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = round(100 * accuracy_score(y_test, model.predict(X_test)), 2)
    st.success(f"Model trained successfully ✅ (Accuracy: {acc}%)")

    st.subheader("🔹 Enter Water Sample Data")

    # Dynamic LOV-style inputs
    user_input = {}
    for col in X.columns:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())

        # Generate 10 evenly spaced values for dropdown
        lov_values = list(pd.Series(pd.Series([min_val, max_val]).astype(float).round(2)))
        lov_values = list(pd.Series(pd.Series([min_val, mean_val, max_val]).astype(float).round(2)))
        
        # Add extra mid-points for better choice
        step_values = list(pd.Series([min_val, (min_val+mean_val)/2, mean_val, (mean_val+max_val)/2, max_val]).round(2))

        user_input[col] = st.selectbox(f"{col}", step_values, index=2)

    # Predict button
    if st.button("Predict Potability"):
        sample_df = pd.DataFrame([user_input])
        prediction = model.predict(sample_df)[0]

        if prediction == 1:
            st.success("💧 Water is **Potable** ✅")
        else:
            st.error("💧 Water is **Not Potable** ❌")
