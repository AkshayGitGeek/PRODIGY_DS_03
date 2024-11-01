---

# Decision Tree Classifier for Predicting Customer Purchases

This project aims to build a Decision Tree Classifier that predicts whether a customer will purchase a product or service based on demographic and behavioral data. Using the **Bank Marketing Dataset** from the UCI Machine Learning Repository, we analyze customer interactions and features to build a model that can predict customer intent effectively.

## Project Overview

In the banking industry, understanding customer behavior and predicting their purchasing intent is essential for targeted marketing and customer engagement strategies. This project utilizes a Decision Tree Classifier to predict whether a customer will subscribe to a term deposit (the target variable) based on demographic and behavioral data.

## Dataset

The **Bank Marketing Dataset** from the UCI Machine Learning Repository is used for this project. This dataset includes details on customer demographics, past interactions with the bank, and other relevant features. The dataset has a target column, `y`, which indicates whether the client subscribed to a term deposit (`yes` or `no`).

### Dataset Details

- **Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **File**: `bank.csv`
- **Number of Instances**: 45,211
- **Number of Attributes**: 17, including both categorical and numerical variables

### Target Variable

- `y`: Indicates whether the client has subscribed to a term deposit (binary classification, `yes` or `no`).

### Feature Variables (Examples)

- **Demographic Attributes**: `age`, `job`, `marital`, `education`
- **Behavioral Attributes**: `campaign` (number of contacts performed), `pdays` (days since the client was last contacted)
- **Other Attributes**: `balance`, `housing` (whether the client has a housing loan), `loan` (personal loan status)

## Project Steps

### 1. Data Preparation

1. **Data Loading**: Load the CSV file and explore the structure.
2. **Data Cleaning**: Handle missing values, if any, and ensure data types are correctly formatted.
3. **Data Encoding**: Convert categorical variables to numeric values using label encoding, as required for the Decision Tree Classifier.
4. **Feature Selection**: Define feature variables (`X`) and the target variable (`y`).

### 2. Model Training and Evaluation

1. **Split the Data**: Divide the data into training and testing sets (e.g., 70% training, 30% testing).
2. **Model Training**: Use a Decision Tree Classifier and fit it to the training data.
3. **Hyperparameter Tuning**: Perform grid search with cross-validation to optimize model parameters, such as `max_depth` and `min_samples_split`.
4. **Model Evaluation**: Assess the model’s performance using metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.

### 3. Feature Importance Analysis

Analyze the feature importance provided by the Decision Tree to understand which factors most influence customer purchasing behavior. Visualize the feature importance to gain insights.

### 4. Results and Findings

Present model performance and key findings, including:
- Model accuracy on the test set.
- Insights from feature importance analysis.
- Suggestions for improving customer engagement based on model predictions.

## Code Example

Below is an example of how the classifier is implemented. Please refer to the main code file for the complete implementation.

```python
# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('path_to_file.csv', sep=';')

# Preprocess and encode categorical data
# Define X (features) and y (target)
X = df.drop('y', axis=1)
y = df['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

## Installation and Requirements

### Prerequisites

- Python 3.6+
- Jupyter Notebook (optional, for interactive work)

### Required Libraries

Install the required Python libraries:

```bash
pip install pandas scikit-learn seaborn matplotlib
```

## Conclusion

This project demonstrates how to use a Decision Tree Classifier to predict customer purchases based on demographic and behavioral data. By tuning model parameters and analyzing feature importance, we can gain valuable insights into customer behavior that can guide marketing and engagement strategies.

---

This README serves as an overview of the project's objective, steps, and implementation approach. It provides clear guidance for replicating the classifier and obtaining similar results.
