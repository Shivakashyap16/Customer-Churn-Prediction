# Customer Churn Prediction & Retention Analytics 📊

Welcome to the **Customer Churn Prediction & Retention Analytics** project! This project uses Machine Learning to identify households that are likely to stop purchasing based on their historical shopping behavior. Built using Python, Scikit-learn, Pandas, and Streamlit, it provides an interactive dashboard to analyze customer churn risk, explore household profiles, and support data-driven customer retention decisions.

The project uses the **Dunnhumby – The Complete Journey dataset** and applies Logistic Regression to predict household-level churn.

# 🛠️ Setup Instructions

Follow these steps to get the project running on your local machine.

### Prerequisites

- Python 3.x installed on your system.
- JupyterLab installed through the project requirements.
- Required Python libraries installed.
- A modern web browser.
- Internet connection to download the dataset and dependencies.

### Steps

**1. Clone this repository to your local machine:**

```bash
git clone https://github.com/Shivakashyap16/Customer-Churn-Prediction.git
```

**2. Navigate to the project directory:**

```bash
cd Customer-Churn-Prediction
```

**3. Download the dataset:**

This project uses the **Dunnhumby – The Complete Journey** retail transaction dataset.

You can download it from either of the following sources:

- Official Dunnhumby source: https://www.dunnhumby.com/source-files/
- Kaggle dataset: https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey

On the official website, locate **The Complete Journey** dataset and follow the download instructions. Alternatively, download the dataset archive from Kaggle.

Extract the downloaded archive. The dataset contains CSV files such as:

- `transaction_data.csv`
- `hh_demographic.csv`
- `product.csv`
- `campaign_desc.csv`
- `campaign_table.csv`
- `causal_data.csv`
- `coupon.csv`
- `coupon_redempt.csv`

**Note:** The current project primarily uses `transaction_data.csv` for data inspection, feature engineering, churn labeling, and model training. The other CSV files are not required for the current prediction workflow.

**4. Place the downloaded dataset in the project folder:**

Create the following folder structure:

```text
Customer-Churn-Prediction/
│
└── data/
    └── raw/
        └── transaction_data.csv
```

Place `transaction_data.csv` directly inside the `data/raw/` folder.

Make sure the file path is:

```text
data/raw/transaction_data.csv
```

The original dataset is intentionally not included in this GitHub repository because of its size. Download it separately and place it in the folder specified above.

**5. Create and activate a virtual environment:**

```bash
python -m venv .venv
```

For Windows:

```bash
.venv\Scripts\activate
```

**6. Install the required dependencies:**

```bash
pip install -r requirements.txt
```

**7. Prepare the dataset and train the model:**

Navigate to the notebooks folder:

```bash
cd notebooks
```

Launch the JupyterLab notebook:

```bash
jupyter lab 01_data_inspection.ipynb
```

Run the notebook cells to perform the following operations:

- Inspect and explore the transaction dataset.
- Analyze household purchasing behavior.
- Define churn based on future purchase activity.
- Perform household-level feature engineering.
- Create the processed household dataset.
- Split the data into training and testing sets.
- Train and evaluate the Logistic Regression model.
- Select the classification threshold using validation data.
- Save the final trained model.

**Important:** In the notebook, skip the older evaluation cell immediately before the cell that defines `final_model`. Run the subsequent cell that defines and trains `final_model`, followed by the final model-saving cell.

The notebook generates the following files automatically:

```text
data/
└── processed/
    └── household_churn_dataset.csv

models/
└── churn_model.joblib
```

The `models/test_metrics.json` file is already included in the repository and contains the held-out test evaluation results displayed by the dashboard.

**8. Return to the project root directory:**

```bash
cd ..
```

**9. Run the Streamlit application:**

```bash
streamlit run dashboard/app.py
```

**10. Open the local URL displayed in your terminal (usually http://localhost:8501) to access the dashboard.**

# 🚀 Dashboard Features

- **Churn Risk Prediction:** Predicts the probability of household churn using a trained Logistic Regression model.
- **Risk Classification:** Categorizes households into Low, Medium, and High risk groups.
- **Customer Profile Insights:** Analyzes purchasing behavior, spending patterns, basket counts, and purchase recency.
- **Household Search & Filtering:** Search household IDs and filter customers based on their churn risk.
- **Household Drill-Down:** Explore individual household purchase profiles and predicted churn probabilities.
- **CSV Export:** Download filtered household churn predictions for further analysis.
- **Model Evaluation:** Displays Precision, Recall, F1-score, PR-AUC, and the confusion matrix using the held-out test set.

# 🖼️ Dashboard Screenshots

<img width="1491" height="780" alt="Screenshot (7)" src="https://github.com/user-attachments/assets/11ce5c38-d94c-4983-916b-f7b60eb54ec4" />

<img width="1466" height="786" alt="Screenshot (8)" src="https://github.com/user-attachments/assets/0ec4ea39-66c3-43dc-89f7-7367b9912894" />

<img width="1459" height="775" alt="Screenshot (9)" src="https://github.com/user-attachments/assets/68683949-9a4f-4f03-b64f-430d43e69c9b" />

# 📈 Model Performance

The Logistic Regression model was evaluated using a separate held-out test set of 500 households.

| Evaluation Metric | Result |
|---|---:|
| Precision | 26% |
| Recall | 63% |
| F1-score | 0.37 |
| PR-AUC | 0.233 |

The model uses a selected classification threshold to identify households at risk of churn. The threshold was selected using validation data, while the reported performance was measured on the held-out test set.

### Confusion Matrix

|  | Predicted: Did Not Churn | Predicted: Churned |
|---|---:|---:|
| **Actual: Did Not Churn** | 402 | 63 |
| **Actual: Churned** | 13 | 22 |

# 🔧 Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn, Logistic Regression
- **Data Processing:** Pandas, NumPy
- **Dashboard:** Streamlit
- **Model Storage:** Joblib
- **Data Analysis:** JupyterLab
- **Data Format:** CSV, JSON
- **Dataset:** Dunnhumby – The Complete Journey

# 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/                  # Downloaded dataset (not included)
│   └── processed/            # Generated processed dataset
│
├── models/
│   ├── churn_model.joblib    # Generated trained model
│   └── test_metrics.json     # Held-out test results
│
├── notebooks/
│   └── 01_data_inspection.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```
