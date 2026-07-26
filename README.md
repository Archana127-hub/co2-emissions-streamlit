# 🚗 CO₂ Emissions Prediction using Machine Learning

## 📌 Project Overview

This project predicts the **CO₂ emissions** of vehicles using a Machine Learning Regression model. The application is built with **Streamlit**, allowing users to enter vehicle specifications and instantly receive predicted CO₂ emission values.

The model is trained on a dataset containing various vehicle characteristics such as engine size, cylinders, fuel consumption, transmission type, and fuel type.

---

## 🎯 Business Objective

The primary objective of this project is to estimate vehicle CO₂ emissions based on engine and fuel consumption features. This helps understand how different vehicle characteristics contribute to carbon emissions and supports environmentally conscious decision-making.

---

## 📂 Dataset

The dataset contains **7,385 vehicle records** with **12 features**.

### Features

- Make
- Model
- Vehicle Class
- Engine Size
- Cylinders
- Transmission
- Fuel Type
- Fuel Consumption (City)
- Fuel Consumption (Highway)
- Fuel Consumption (Combined L/100 km)
- Fuel Consumption (Combined MPG)

### Target Variable

- CO₂ Emissions (g/km)

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Model Deployment using Streamlit

---

## 🚀 Features

- Interactive Streamlit interface
- Real-time CO₂ emission prediction
- Easy-to-use input fields
- Fast prediction using a trained ML model
- Simple and responsive UI

---

## 📁 Project Structure

```
CO2-Emissions-Prediction/
│
├── app.py
├── co2_model.pkl
├── scaler.pkl
├── requirements.txt
├── dataset.csv
├── README.md
└── images/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/co2-emissions-prediction.git
```

Move into the project folder:

```bash
cd co2-emissions-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📈 Model Performance

The trained regression model predicts vehicle CO₂ emissions based on engine specifications and fuel consumption values.

Evaluation metrics may include:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## 💻 Streamlit Application

The web application allows users to:

- Enter vehicle details
- Click the **Predict** button
- View the predicted CO₂ emission value instantly

---

## 🔮 Future Improvements

- Deploy the application online
- Add more regression models for comparison
- Improve prediction accuracy through hyperparameter tuning
- Visualize prediction results with charts
- Support batch predictions using CSV upload

---

## 👩‍💻 Author

**Gaddam Archana**

B.Tech Graduate

Machine Learning Project

---

## 📄 License

This project is developed for educational and learning purposes.
