# House-Price-Predictor
A machine learning model that predicts house prices based on property features, trained on Indian housing data.
## Repository structure
```
|   .gitignore
|   app.py
|   project_structure.txt
|   requirements.txt
|   
+---data
|   +---processed
|   |       housing_cleaned.csv
|   |       
|   \---raw
|           housing.csv
|           
+---models
|       feature_selector.pkl
|       median_pps.pkl
|       train_model.pkl
|       
\---src
        clean_data.py
        model.py
```

## Business problem
The goal of the project was to develop an application that can quickly predict the price of housing given the parameters
## Tech Stack
- **Python 3.x** – core language  
- **Streamlit** – web UI  
- **scikit-learn** – modeling  
- **pandas** – data manipulation  
- **NumPy** – numerical operations  
## Data Source
[kaggle housing prices dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
## Main Features
- Predicts house prices using Gradient Boosting Regression
- Interactive web interface with Streamlit
- Handles features like area, bedrooms, bathrooms, luxury score, etc.
- Converts prices from INR to USD for international users
## Run Locally
### **1. Clone the Repository**
```bash
git clone https://github.com/PavelVitAnt/House-Price-Predictor.git
cd House-Price-Predictor
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3. Launch the Streamlit App**
```bash
streamlit run app.py
```

## License
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

© 2025 Pavel Antipov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
