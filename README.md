# 🚦 AI-Based Smart Traffic Congestion Prediction and Route Optimization

## 📌 Project Overview

The **AI-Based Smart Traffic Congestion Prediction and Route Optimization** system is an intelligent transportation solution that uses **Machine Learning** and **Graph Algorithms** to predict traffic congestion and identify traffic-aware routes.

The system analyzes traffic conditions such as vehicle count, average speed, road capacity, weather, accidents, road work, special events, time of day, and road characteristics.

The project provides an interactive **Streamlit dashboard** where users can enter traffic conditions, receive an AI-based congestion prediction, and find an optimized route between locations.

---

## 🎯 Objectives

- Predict traffic congestion levels using Machine Learning.
- Classify traffic into:
  - Low
  - Medium
  - High
  - Severe
- Analyze traffic patterns using Exploratory Data Analysis.
- Compare multiple Machine Learning algorithms.
- Select a realistic production model without target leakage.
- Optimize routes using Dijkstra's algorithm.
- Compare route results with A* search.
- Provide an interactive Streamlit dashboard.
- Present traffic analytics through charts and tables.

---

## 🧠 Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Streamlit

### Route Optimization

- NetworkX
- Dijkstra's Algorithm
- A* Search

### Model Management

- Joblib

---

## 📂 Project Structure

```text
AI-Based Smart Traffic Congestion Prediction and Route Optimization
│
├── data
│   ├── raw
│   │   └── traffic_data.csv
│   │
│   └── processed
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── preprocessor.pkl
│
├── models
│   ├── best_traffic_congestion_model.pkl
│   ├── best_model_name.txt
│   ├── production_traffic_model.pkl
│   └── production_model_name.txt
│
├── results
│   ├── eda
│   ├── models
│   ├── models_v2
│   └── routes
│
├── src
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_models.py
│   ├── train_models_v2.py
│   └── route_optimizer.py
│
├── app.py
├── requirements.txt
└── README.md
