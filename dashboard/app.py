from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Locate project files
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "household_churn_dataset.csv"
METRICS_PATH = PROJECT_ROOT / "models" / "test_metrics.json"

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction & Retention Analytics")
st.caption(
    "Household-level churn predictions using the Dunnhumby Complete Journey dataset"
)


# --------------------------------------------------
# 2. Load saved model, dataset, and test metrics
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_test_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    saved_artifact = load_model()
    data = load_data()
    test_metrics = load_test_metrics()

except FileNotFoundError as error:
    st.error(f"Could not find a required file: {error}")
    st.stop()

except json.JSONDecodeError:
    st.error(
        "The test metrics file is not valid JSON. "
        "Please check models/test_metrics.json."
    )
    st.stop()


model = saved_artifact["model"]
threshold = saved_artifact["threshold"]
feature_columns = saved_artifact["feature_columns"]


# --------------------------------------------------
# 3. Generate household churn scores
# --------------------------------------------------
X = data[feature_columns]

data["churn_probability"] = model.predict_proba(X)[:, 1]

data["predicted_churn"] = (
    data["churn_probability"] >= threshold
).astype(int)

data["risk_band"] = pd.cut(
    data["churn_probability"],
    bins=[-0.001, 0.30, 0.60, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)


# --------------------------------------------------
# 4. Summary metrics
# --------------------------------------------------
total_households = len(data)
high_risk = int(data["predicted_churn"].sum())
average_risk = data["churn_probability"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Households analyzed", f"{total_households:,}")
col2.metric("Flagged by model threshold", f"{high_risk:,}")
col3.metric("Average predicted churn risk", f"{average_risk:.1%}")

st.write(f"**Model decision threshold:** {threshold:.3f}")


# --------------------------------------------------
# 5. Risk-band breakdown
# --------------------------------------------------
st.subheader("Churn risk breakdown")

risk_order = ["Low", "Medium", "High"]

risk_summary = (
    data.groupby("risk_band", observed=False)
    .agg(
        households=("household_key", "count"),
        average_predicted_risk=("churn_probability", "mean"),
        historical_churn_rate=("churn", "mean")
    )
    .reindex(risk_order)
    .fillna(0)
)

risk_summary["household_percentage"] = (
    risk_summary["households"] / total_households * 100
)

risk_summary = risk_summary.rename(
    columns={
        "households": "Households",
        "household_percentage": "Share of households (%)",
        "average_predicted_risk": "Average predicted risk",
        "historical_churn_rate": "Historical churn rate"
    }
)

left, right = st.columns(2)

with left:
    st.caption("Number of households by risk band")
    st.bar_chart(risk_summary["Households"])

with right:
    st.caption("Share of households by risk band")
    st.bar_chart(risk_summary["Share of households (%)"])

st.caption(
    "Risk bands are dashboard groupings: Low ≤30%, Medium >30% to 60%, "
    "High >60%. Historical churn rate is based on the dataset's known labels."
)

st.dataframe(
    risk_summary.style.format({
        "Share of households (%)": "{:.1f}%",
        "Average predicted risk": "{:.1%}",
        "Historical churn rate": "{:.1%}"
    }),
    use_container_width=True
)


# --------------------------------------------------
# 6. Customer profile insights
# --------------------------------------------------
st.subheader("Customer profile insights")

profile = (
    data.groupby("predicted_churn", observed=False)
    .agg(
        households=("household_key", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_baskets=("basket_count", "mean"),
        avg_total_spend=("total_spend", "mean"),
        avg_unique_products=("unique_products", "mean")
    )
    .rename(index={0: "Lower risk", 1: "Flagged high risk"})
    .round(2)
)

st.dataframe(profile, use_container_width=True)


# --------------------------------------------------
# 7. Household risk table, filters, search, and CSV
# --------------------------------------------------
st.subheader("Household churn risk")

risk_filter = st.selectbox(
    "Filter households by risk band",
    ["All households", "High-risk only", "Medium-risk only", "Low-risk only"]
)

search_value = st.text_input(
    "Search household ID",
    placeholder="Enter a household ID, e.g. 123"
).strip()

if risk_filter == "High-risk only":
    displayed_data = data[data["risk_band"] == "High"]
elif risk_filter == "Medium-risk only":
    displayed_data = data[data["risk_band"] == "Medium"]
elif risk_filter == "Low-risk only":
    displayed_data = data[data["risk_band"] == "Low"]
else:
    displayed_data = data

if search_value:
    displayed_data = displayed_data[
        displayed_data["household_key"]
        .astype(str)
        .str.contains(search_value, case=False, na=False)
    ]

display_columns = [
    "household_key",
    "risk_band",
    "churn_probability",
    "predicted_churn",
    "recency_days",
    "basket_count",
    "total_spend",
    "unique_products"
]

display_data = (
    displayed_data[display_columns]
    .sort_values("churn_probability", ascending=False)
    .rename(columns={
        "household_key": "Household",
        "risk_band": "Risk band",
        "churn_probability": "Churn risk",
        "predicted_churn": "Flagged by threshold",
        "recency_days": "Days since last purchase",
        "basket_count": "Number of baskets",
        "total_spend": "Total spend",
        "unique_products": "Unique products"
    })
)

st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)

csv_data = display_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download displayed households as CSV",
    data=csv_data,
    file_name="household_churn_scores.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 8. Household drill-down
# --------------------------------------------------
st.subheader("Household drill-down")

household_ids = sorted(data["household_key"].unique())

selected_household = st.selectbox(
    "Select a household to inspect",
    household_ids
)

selected_row = data.loc[
    data["household_key"] == selected_household
].iloc[0]

risk_probability = selected_row["churn_probability"]
risk_label = selected_row["risk_band"]
is_flagged = selected_row["predicted_churn"] == 1

detail_col1, detail_col2, detail_col3 = st.columns(3)

detail_col1.metric("Predicted churn risk", f"{risk_probability:.1%}")
detail_col2.metric("Risk band", str(risk_label))
detail_col3.metric(
    "Model threshold result",
    "Flagged" if is_flagged else "Not flagged"
)

st.write("**Household purchase profile**")

feature_labels = {
    "last_purchase_day": "Last purchase day",
    "basket_count": "Number of baskets",
    "total_spend": "Total spend",
    "total_items": "Total items purchased",
    "unique_products": "Unique products purchased",
    "active_days": "Number of active purchase days",
    "recency_days": "Days since last purchase"
}

detail_rows = []

for feature in feature_columns:
    if feature in selected_row.index:
        detail_rows.append({
            "Feature": feature_labels.get(feature, feature),
            "Value": selected_row[feature]
        })

st.dataframe(
    pd.DataFrame(detail_rows),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 9. Historical outcome comparison
# --------------------------------------------------
st.subheader("Historical outcome comparison")

st.info(
    "These are household-level predictions on the full historical dataset. "
    "They are useful for exploring the dashboard, but they are not the final "
    "held-out test evaluation because some households were used during model training."
)

if "churn" in data.columns:
    outcome_table = (
        data.groupby(["risk_band", "churn"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(risk_order)
        .rename(columns={0: "Did not churn", 1: "Churned"})
    )

    st.dataframe(outcome_table, use_container_width=True)


# --------------------------------------------------
# 10. Held-out test model evaluation
# --------------------------------------------------
st.subheader("Model evaluation — held-out test set")

st.caption(
    f"These metrics come from a separate test set of "
    f"{test_metrics['test_set_size']:,} households, "
    "not from the full-dataset dashboard predictions."
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Precision",
    f"{test_metrics['precision']:.0%}"
)
metric_col2.metric(
    "Recall",
    f"{test_metrics['recall']:.0%}"
)
metric_col3.metric(
    "F1-score",
    f"{test_metrics['f1_score']:.2f}"
)
metric_col4.metric(
    "PR-AUC",
    f"{test_metrics['pr_auc']:.3f}"
)

st.write("**Confusion matrix**")

confusion = test_metrics["confusion_matrix"]

confusion_matrix = pd.DataFrame(
    [
        [
            confusion["true_negative"],
            confusion["false_positive"]
        ],
        [
            confusion["false_negative"],
            confusion["true_positive"]
        ]
    ],
    index=["Actual: did not churn", "Actual: churned"],
    columns=["Predicted: did not churn", "Predicted: churned"]
)

st.dataframe(confusion_matrix, use_container_width=True)

st.caption(
    f"At the selected threshold, the model identified "
    f"{confusion['true_positive']} of "
    f"{confusion['true_positive'] + confusion['false_negative']} churners "
    "in the held-out test set, while "
    f"{confusion['false_positive']} non-churning households "
    "were incorrectly flagged."
)


# --------------------------------------------------
# 11. Limitations
# --------------------------------------------------
st.subheader("Limitations")

st.markdown("""
- **Historical data:** The dashboard uses historical household transactions.
  Predictions may not represent current customer behavior.
- **Project-defined churn:** Churn is defined as no purchase during the
  90-day prediction window. Other businesses may define churn differently.
- **Retention actions:** Any suggested retention actions would be
  rule-based ideas, not validated interventions. Their effectiveness
  would need to be tested.
- **Evaluation context:** The dashboard scores the full historical dataset
  for exploration. Use the held-out test metrics above to report model
  performance.
""")