import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv(R"C:\Users\pavel\Documents\archive\Housing.csv")

# 2. Convert 'yes'/'no' columns to binary (1/0)
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                 'airconditioning', 'prefarea']

for col in binary_columns:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# 3. Convert furnishingstatus to ordinal
furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
df['furnishingstatus'] = df['furnishingstatus'].map(furnishing_map)



# 4. Handle any potential missing values
df = df.dropna()  


# 5. Create a total rooms feature
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# 6. Create a luxury score
df['luxury_score'] = (df['airconditioning'] + df['prefarea'] + 
                     df['guestroom'] + df['hotwaterheating'])


# 7. Save cleaned dataset
df.to_csv('Housing_Cleaned.csv', index=False)