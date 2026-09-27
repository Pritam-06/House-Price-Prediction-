# 🏠 House Price Prediction using Machine Learning

A machine learning project that predicts **median house prices** using housing-related features such as location, number of rooms, population, household information, median income, and ocean proximity.

The project uses **Python, Pandas, NumPy, Scikit-learn, and Joblib**, with a complete preprocessing and machine learning pipeline.

---   

## 📌 Project Overview

House prices depend on several factors such as location, income level, number of rooms, population, and proximity to the ocean.

In this project, a **Random Forest Regression** model is trained on the California housing dataset to predict the `median_house_value` of a property.

The project also demonstrates an important real-world machine learning workflow:

**Dataset → Data Splitting → Preprocessing → Feature Transformation → Model Training → Model Saving → Prediction**

---

## 🎯 Objectives

* Understand the complete machine learning workflow.
* Prepare a dataset for machine learning.
* Perform a stratified train/test split.
* Handle missing numerical values.
* Scale numerical features.
* Encode categorical features.
* Build a reusable Scikit-learn preprocessing pipeline.
* Train a Random Forest Regression model.
* Save the trained model and preprocessing pipeline.
* Use the saved model to make predictions on new data.
* Export predictions to a CSV file.

---

## 🗂️ Project Structure

```text
House-Price-Prediction/
│
├── housing.csv          # Main housing dataset
├── input.csv            # Test/input data used for prediction
├── output.csv           # Generated predictions
│
├── main2.py              # Main training and prediction program
├── main1.py          # Earlier version of the implementation
│
├── model.pkl            # Saved trained machine learning model
├── pipeline.pkl         # Saved preprocessing pipeline
│
├── input - Copy.csv     # Copy of input dataset
│
└── README.md            # Project documentation
```

> `model.pkl` and `pipeline.pkl` are generated automatically after the model-training stage.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning and preprocessing
* **Joblib** – Saving and loading trained models
* **CSV** – Dataset and prediction storage

---

## 📊 Dataset

The project uses the **California Housing Dataset**, containing information about different housing districts.

### Important Features

| Feature              | Description                                     |
| -------------------- | ----------------------------------------------- |
| `longitude`          | Longitude of the district                       |
| `latitude`           | Latitude of the district                        |
| `housing_median_age` | Median age of houses                            |
| `total_rooms`        | Total number of rooms                           |
| `total_bedrooms`     | Total number of bedrooms                        |
| `population`         | Population of the district                      |
| `households`         | Number of households                            |
| `median_income`      | Median income of households                     |
| `ocean_proximity`    | Proximity of the district to the ocean          |
| `median_house_value` | Target variable representing median house value |

The target variable is:

```text
median_house_value
```

---

## 🔄 Machine Learning Workflow

### 1. Load the Dataset

The housing dataset is loaded using Pandas:

```python
housing = pd.read_csv("housing.csv")
```
    
---

### 2. Stratified Train/Test Split

The project creates an `income_cat` feature based on `median_income` and uses `StratifiedShuffleSplit`.

This helps maintain a similar distribution of income categories between the training and testing datasets.

```python
housing['income_cat'] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)
```

The split uses:

```python
test_size=0.2
random_state=42
```

Therefore, approximately **80% of the data is used for training and 20% for testing/inference input**.

---

## 🧹 Data Preprocessing

The project separates numerical and categorical features.

### Numerical Features

Numerical columns are processed using:

1. **Median imputation** for missing values.
2. **StandardScaler** for feature scaling.

```python
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

### Categorical Features

The `ocean_proximity` column is categorical and is converted using One-Hot Encoding:

```python
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---

## 🔗 Preprocessing Pipeline

A `ColumnTransformer` combines the numerical and categorical preprocessing steps:

```python
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])
```

This makes preprocessing consistent between model training and prediction.

---

## 🤖 Machine Learning Model

The final implementation uses:

### Random Forest Regressor

```python
model = RandomForestRegressor(random_state=42)
model.fit(housing_prepared, housing_labels)
```

Random Forest combines multiple decision trees to perform regression.

The model predicts:

```text
median_house_value
```

---

## 💾 Model Saving

After training, the model and preprocessing pipeline are saved using Joblib.

```python
joblib.dump(model, "model.pkl")
joblib.dump(pipeline, "pipeline.pkl")
```

This allows the trained model to be reused without training it again every time.

---

## 🔮 Making Predictions

When `model.pkl` already exists, `main.py` loads the saved model and pipeline:

```python
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)
```

The input data is then transformed:

```python
transformed_input = pipeline.transform(input_data)
```

Predictions are generated:

```python
predictions = model.predict(transformed_input)
```

The predicted values are added to the input dataset and saved as:

```text
output.csv
```

---

## 📈 Example Prediction

An example prediction generated by the project:

```text
Input:
Longitude: -118.39
Latitude: 34.12
Median Income: 8.2816
Ocean Proximity: <1H OCEAN

Predicted Median House Value:
483020.60
```

The prediction value is generated by the trained Random Forest model and is not a guaranteed market value.

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```

Move into the project directory:

```bash
cd House-Price-Prediction
```

### 2. Install Required Libraries

```bash
pip install pandas numpy scikit-learn joblib
```

### 3. Run the Program

```bash
python main.py
```

### 4. First Run

If `model.pkl` does not exist, the program will:

```text
Load Dataset
      ↓
Create Stratified Split
      ↓
Preprocess Data
      ↓
Train Random Forest Model
      ↓
Save model.pkl
      ↓
Save pipeline.pkl
```

### 5. Subsequent Run

If the saved model already exists, the program performs inference:

```text
Load model.pkl
      ↓
Load pipeline.pkl
      ↓
Read input.csv
      ↓
Transform Input
      ↓
Predict House Prices
      ↓
Create output.csv
```

---

## 📁 Input and Output

### Input

The prediction data is stored in:

```text
input.csv
```

It contains the housing features without the target variable.

### Output

Predictions are stored in:

```text
output.csv
```

The output contains the original input features along with:

```text
median_house_value
```

---

## 🧠 What I Learned From This Project

This project helped me understand several important machine learning concepts:

* Data loading using Pandas
* Feature and target separation
* Stratified sampling
* Numerical data preprocessing
* Handling missing values
* Feature scaling
* Categorical encoding
* Scikit-learn pipelines
* ColumnTransformer
* Random Forest Regression
* Model persistence using Joblib
* Training vs inference
* Making predictions on unseen data
* Working with CSV datasets

---

## 🚀 Future Improvements

The project can be extended by adding:

* Exploratory Data Analysis (EDA)
* Data visualization
* Feature engineering
* Model comparison
* Cross-validation
* Hyperparameter tuning
* RMSE and MAE evaluation on the test set
* Prediction interface using Streamlit
* Web-based prediction application
* Model performance visualizations
* Better project documentation

Possible models for comparison:

```text
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting
```

---

## 📌 Project Status

**Status:** Completed – Basic Machine Learning Regression Project

The current version focuses on understanding the fundamental machine learning workflow and implementing a reusable preprocessing and prediction pipeline.

---

## 👨‍💻 Author

**Pritam Chavhan**

Electronics & Telecommunication Engineering Student
Interested in Data Science, Machine Learning and Python.

---

## ⭐ Acknowledgement

This project was created as a learning project to understand the practical implementation of a machine learning regression workflow using Python and Scikit-learn.

---

## 📄 License

This project is intended for educational and learning purposes.
