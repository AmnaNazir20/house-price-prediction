# House-Price-Prediction
A machine learning project that predicts house prices using the California Housing Dataset. Includes data exploration, preprocessing (log transformation, hot encoding, scaling), and model training using Linear Regression and Random Forest. Evaluated using RMSE and R² metrics.

---

## 🚀 Project Overview


This project uses the **California Housing Dataset** to predict house prices using **Machine Learning** models. The goal is to analyze housing features and build accurate predictive models using **Linear Regression** and **Random Forest Regressor**,comparing their performances using RMSE and R² metrics

---

## 📊 Dataset

- **Dataset**: California Housing Prices  
- **Source**: [Kaggle – California Housing Prices Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)  
- **Total Samples**: 20,640  
- **Total Features**: 10  
- **Target Variable**: `median_house_value`
---

##  Data Exploration & Preprocessing

### 📊 Dataset Dimensions
- **Rows:** 20,640  
- **Columns:** 10

---

### 🕵️ Initial Inspection
- Verified data types of each column.
- Checked for null or missing values.
- Confirmed no duplicate rows in the dataset.

---

### 🧠 Feature Engineering

- Applied **log transformation** to reduce skewness in left- and right-skewed features.
- Transformed distributions to be more **bell-shaped**, enhancing model performance.

![Log Transformed Histogram](images/log_transformation_hist.png)

---

### 🔤 Categorical Encoding
- Applied **One-Hot Encoding** to convert the categorical `ocean_proximity` column into numeric format.

---

### 📈 Correlation Analysis

- Generated a **correlation matrix** to visualize relationships among features.
- Selected variables that showed strong correlation with the target variable `median_house_value`.

![Correlation Matrix](images/correlation_matrix.png)


### 🧪 Train-Test Split
- Split the dataset into:
  - **Training set:** 80%
  - **Testing set:** 20%

---

### 📐 Feature Scaling
- Applied **StandardScaler** to normalize the data.
- Performed scaling **after splitting** the data to avoid data leakage.
- Scaled numerical features separately for training and testing sets.

---
## 🛠️ Tech Stack

- Python
- NumPy, Pandas
- Scikit-learn (sklearn)
- Matplotlib, Seaborn


---

##  Models Used

### 1️⃣ Linear Regression

A simple baseline model to capture linear relationships between features and the target variable.


### 2️⃣ Random Forest Regressor
- An ensemble tree-based model that handles non-linearities better
- 📉 Evaluated using:
  - RMSE (Root Mean Squared Error)
  - R² Score
---

##  Model Comparison

| Model             | RMSE         | R² Score   |
|------------------|--------------|------------|
| Linear Regression| 69,356.32    | 0.63       |
| Random Forest    | 48,960.55    | 0.82       |

✅ **Random Forest Regressor outperformed Linear Regression** significantly.

---

## 📁 Project Structure

```
├── data/
│ └── housing.csv # Raw dataset
├── notebooks/
│ └── house_price_prediction.ipynb 
└── README.md # Project overview and documentation
```

## 💻 How to Run

```bash
git clone https://github.com/AmnaNazir20/house-price-prediction.git
cd spam-email-classifier
pip install -r requirements.txt
```

---
## 🧠 Future Improvements

- Add hyperparameter tuning (GridSearchCV)
- Try other regressors (Gradient Boosting, XGBoost)
- Deploy model using Flask or Streamlit

---

## 🙋‍♀️ Author

**Amna Nazir**  
🎓 MS Data Science | FAST University  
🔗 [LinkedIn Profile](www.linkedin.com/in/amna-nazir-460b1936a)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
