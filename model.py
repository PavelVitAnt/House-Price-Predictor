import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
import pickle
import os

def train_model():
    # Load data
    df = pd.read_csv("Housing_Cleaned.csv")
    
    # Feature selection - keep only these impactful features
    core_features = [
        'area',
        'bedrooms',
        'bathrooms',
        'stories',
        'luxury_score',
        'total_rooms',
        'airconditioning',
        'prefarea'
    ]
    
    # Create one useful engineered feature
    df['price_per_sqft'] = df['price'] / df['area']
    
    # Final feature set
    X = df[core_features + ['price_per_sqft']]
    y = df['price']
    
    # Optimized Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
    
    # Feature selection wrapper
    selector = SelectFromModel(model, threshold='median')
    X_selected = selector.fit_transform(X, y)
    
    # Train final model on selected features
    model.fit(X_selected, y)

    # Find median price per sqft
    median_price_per_sqft = (df['price'] / df['area']).median()
    
    return model, selector, median_price_per_sqft

def predict_price(model, selector, median_price_per_sqft):
    print("\nEnter the following details to predict the house price:")
    
    # Get user input for each feature
    area = float(input("Area (in sqft): "))
    bedrooms = int(input("Number of bedrooms: "))
    bathrooms = int(input("Number of bathrooms: "))
    stories = int(input("Number of stories: "))
    luxury_score = int(input("Luxury score (0-3): "))
    total_rooms = int(input("Total rooms: "))
    airconditioning = int(input("Air conditioning (1 for yes, 0 for no): "))
    prefarea = int(input("Preferred area (1 for yes, 0 for no): "))
    
    
    # Create feature array
    features = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        luxury_score,
        total_rooms,
        airconditioning,
        prefarea,
        median_price_per_sqft  
    ]])
    
    # Select features (same as during training)
    features_selected = selector.transform(features)
    
    # Make prediction
    predicted_price = model.predict(features_selected)[0]
    
    # Update price_per_sqft with the predicted price
    updated_price_per_sqft = predicted_price / area
    features[0, -1] = updated_price_per_sqft
    
    # Make final prediction with updated price_per_sqft
    features_selected = selector.transform(features)
    predicted_price = model.predict(features_selected)[0]
    
    print(f"\nPredicted House Price: ₹{predicted_price:,.2f}")

def main():
    # Train or load the model
    print("Loading the prediction model...")
    model, selector, median_pps = train_model()
    
    while True:
        predict_price(model, selector, median_pps)
        
        # Ask if user wants to make another prediction
        another = input("\nWould you like to make another prediction? (yes/no): ").lower()
        if another != 'yes':
            print("\nThank you for using the House Price Predictor!")
            break

if __name__ == "__main__":
    
    # Save model
    os.makedirs('model', exist_ok=True)
    
    model, selector, median_pps = train_model()

    with open('model/train_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/feature_selector.pkl', 'wb') as f:
        pickle.dump(selector, f)
    with open('model/median_pps.pkl', 'wb') as f:
        pickle.dump(median_pps, f)

    main()