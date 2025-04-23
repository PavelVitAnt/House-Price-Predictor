import streamlit as st
import pickle
import numpy as np

INR_TO_USD = 83.0
# Load the saved model components
@st.cache_resource
def load_model():
    with open('train_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open('median_pps.pkl', 'rb') as f:
        median_pps = pickle.load(f)
    return model, selector, median_pps

model, selector, median_price_per_sqft = load_model()

def predict_price(input_features):
        features = np.array([[
        input_features['area'],
        input_features['bedrooms'],
        input_features['bathrooms'],
        input_features['stories'],
        input_features['luxury_score'],
        input_features['total_rooms'],
        input_features['airconditioning'],
        input_features['prefarea'],
        median_price_per_sqft
    ]])
        features_selected = selector.transform(features)
        predicted_price = model.predict(features_selected)[0]
         
        # Update price_per_sqft and predict again
        updated_price_per_sqft = predicted_price / input_features['area']
        features[0, -1] = updated_price_per_sqft
        features_selected = selector.transform(features)
        predicted_price = model.predict(features_selected)[0]
        
        return predicted_price
def main():
    st.title("House Price Predictor.")
    st.subheader("Important Note: The model was trained on Indian housing data and the prices were later converted to dollars. Therefore, the prices may not accurately reflect the actual housing market conditions in your country")
    st.write("Enter the house details to get a price prediction:")
    # Input fields
    area = st.number_input("Area (in sqft)(300 min, 10000 max)", min_value=300, max_value=10000, value=1000)
    bedrooms = st.number_input("Number of bedrooms(1 min, 10 max)", min_value=1, max_value=10, value=2)
    bathrooms = st.number_input("Number of bathrooms(1 min, 5 max)", min_value=1, max_value=5, value=1)
    stories = st.number_input("Number of stories(1 min, 4 max)", min_value=1, max_value=4, value=1)
    luxury_score = st.slider("Luxury score (0-3)", min_value=0, max_value=3, value=1)
    total_rooms = st.number_input("Total rooms(1 min, 15 max)", min_value=1, max_value=15, value=3)
    airconditioning = st.selectbox("Air conditioning", ["No", "Yes"])
    prefarea = st.selectbox("Preferred area", ["No", "Yes"])

    if st.button("Predict Price"):
        input_features = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'luxury_score': luxury_score,
            'total_rooms': total_rooms,
            'airconditioning': 1 if airconditioning == "Yes" else 0,
            'prefarea': 1 if prefarea == "Yes" else 0
        }
    

        predicted_price = predict_price(input_features)
        
        
        predicted_price_usd = predicted_price / INR_TO_USD
        
        
        st.success(f"Predicted House Price: ${predicted_price_usd:,.2f}")
        

if __name__ == "__main__":
    main()
