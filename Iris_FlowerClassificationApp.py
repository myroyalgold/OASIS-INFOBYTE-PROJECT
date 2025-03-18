import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset 
df = pd.read_csv("Iris.csv")

# drop the 'Id' column
df = df.drop(columns=["Id"])

# encode the 'Species' column
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Split data into features (X) and target (y)
X = df.drop(columns=["Species"])
y = df["Species"]

# Split dataset into training 80% and testing 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

# define a function for predicting iris species
species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    user_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                             columns=X.columns)
    prediction = rf_model.predict(user_data)[0]
    return species_mapping[prediction]

# Streamlit App
st.sidebar.title("IRIS FLOWER CLASSIFICATION")

#three equal columns in the sidebar
col1, col2, col3 = st.sidebar.columns(3)

# images in separate columns to maintain same height
col1.image("Flower_img/setosa.jpeg", width=100)
col1.write("Iris Setosa")
col2.image("Flower_img/versicolor.jpeg", width=100)
col2.write("Iris Versicolor")
col3.image("Flower_img/virginica_lg.jpg", width=100)
col3.write("Iris Virginica")

if st.sidebar.button("RandomForest Accuracy"):
    st.sidebar.write(f"RandomForest Accuracy: {rf_accuracy * 100:.2f}%")

if st.sidebar.button("Logistic Regression Accuracy"):
    st.sidebar.write(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

if st.sidebar.button("Visualize Important Features"):
    feature_importances = rf_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)
    sorted_features = [X.columns[i] for i in sorted_indices]
    sorted_importances = feature_importances[sorted_indices]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis", hue=sorted_features, legend=False)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Flower Features")
    plt.title("Feature Importance for Iris Classification (Sorted)")
    st.pyplot(plt)

st.title("IRIS FLOWER CLASSIFICATION")

sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=3.0, step=0.1)

if st.button("Predict Iris Flower"):
    prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    st.write(f"Predicted Species: {prediction}")