import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# loading model
with open("property_model.pkl", "rb") as file:
    model = pickle.load(file)
    
#streamlit stuff
st.title("Property value trends with AI - Noida")
st.write("ML powered property price prediction for Noida.")

# inputs
size = st.number_input("Please provide the property size (in square feets): ", min_value=200, max_value=5000, value=1000)    # min,max,default
bedrooms = st.slider("Number of bedrooms: ", 1,5,2) # min,max,default
location_score= st.slider("Location Score (1 => poor, 10 => excellent)", 1,10,5)  # min,max,default

if st.button("Predict Price"):
    features = np.array([[size, bedrooms, location_score]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Property price: ₹ {prediction:,.0f}")
    
    #  Price vs Size (line chart)
    size_vals = np.linspace(200, 5000, 20)
    preds_size = model.predict(
        np.column_stack([size_vals, np.full(20, bedrooms), np.full(20, location_score)])
    )
    fig1 = px.line(x=size_vals, y=preds_size, labels={"x": "Size (sqft)", "y": "Price"},
                   title="📈 Price vs Size")

    #  Price vs Bedrooms (bar chart)
    bedroom_vals = np.arange(1, 6)
    preds_bedrooms = model.predict(
        np.column_stack([np.full(5, size), bedroom_vals, np.full(5, location_score)])
    )
    fig2 = px.bar(x=bedroom_vals, y=preds_bedrooms,
                  labels={"x": "Bedrooms", "y": "Price"},
                  title="Price vs Bedrooms")

    #  Heatmap (Size × Bedrooms vs Price)
    grid = []
    for s in size_vals:
        for b in bedroom_vals:
            pred = model.predict([[s, b, location_score]])[0]
            grid.append((s, b, pred))
    df = pd.DataFrame(grid, columns=["Size", "Bedrooms", "Price"])

    fig3 = px.density_heatmap(df, x="Size", y="Bedrooms", z="Price", histfunc="avg",
                              title="Heatmap of Size & Bedrooms vs Price")

    # Layout: 2 columns for first two charts
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

    # Heatmap full width below
    st.plotly_chart(fig3, use_container_width=True)