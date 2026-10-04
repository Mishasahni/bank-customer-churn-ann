"""Bank Customer Churn & Retention dashboard.

Run from the project root:
    streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# Make src/ importable when Streamlit runs this file directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.churn_utils import (  # noqa: E402
    COLOR_MUTED,
    COLOR_PRIMARY,
    FEATURES,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RISK_TIERS,
    TARGET,
    TRAIN_PATH,
    assign_risk_tier,
    churn_rate_table,
    load_raw_data,
)

st.set_page_config(page_title="Bank Churn & Retention", page_icon="🏦", layout="wide")

AGE_BANDS = ["18–29", "30–39", "40–49", "50–59", "60+"]
RISK_BADGES = {"Low": "🟢 LOW", "Medium": "🟠 MEDIUM", "High": "🔴 HIGH"}


# ---------------------------------------------------------------------------
# Data and model loading (cached so they load once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_customers():
    df = load_raw_data().drop(columns=["RowNumber", "Surname"])
    df["AgeBand"] = pd.cut(df["Age"], bins=[17, 29, 39, 49, 59, 120], labels=AGE_BANDS)
    df["Activity"] = df["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    df["CreditCard"] = df["HasCrCard"].map({0: "No card", 1: "Has card"})
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(MODEL_METADATA_PATH.read_text())


@st.cache_data
def load_balance_cutoff():
    """Median training-set balance: the 'higher value' cut-off used in notebook 05."""
    return pd.read_csv(TRAIN_PATH)["Balance"].median()


def churn_bar_chart(data, column, title, sort=None, overall_rate=None):
    """Churn-rate bar chart with a dashed line at the overall churn rate."""
    table = churn_rate_table(data, column)
    table[column] = table[column].astype(str)
    base = alt.Chart(table).encode(
        x=alt.X(f"{column}:N", sort=sort, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("churn_rate:Q", title="Churn rate (%)"),
        tooltip=[alt.Tooltip(f"{column}:N", title=column),
                 alt.Tooltip("churn_rate:Q", title="Churn rate (%)", format=".1f"),
                 alt.Tooltip("customers:Q", title="Customers", format=",")],
    )
    bars = base.mark_bar(color=COLOR_PRIMARY, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=38)
    labels = base.mark_text(dy=-8, fontWeight="bold").encode(text=alt.Text("churn_rate:Q", format=".1f"))
    chart = bars + labels
    if overall_rate is not None:
        rule = alt.Chart(pd.DataFrame({"rate": [overall_rate]})).mark_rule(
            strokeDash=[5, 4], color=COLOR_MUTED).encode(y="rate:Q")
        chart = chart + rule
    return chart.properties(title=title, height=280)


customers = load_customers()
model, metadata = load_model()
overall_churn = customers[TARGET].mean() * 100

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏦 Churn & Retention")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Customer analysis", "Churn risk predictor", "Model performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: 10,000 bank customers (France, Germany, Spain). "
    "Extends the open-source project "
    "[sinanw/ann-bank-customer-retention](https://github.com/sinanw/ann-bank-customer-retention)."
)

# ---------------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Bank Customer Churn Overview")
    st.caption("Who leaves the bank, and which customer groups show the highest churn?")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(customers):,}")
    c2.metric("Customers churned", f"{customers[TARGET].sum():,}")
    c3.metric("Overall churn rate", f"{overall_churn:.1f}%")
    c4.metric("Active members", f"{customers['IsActiveMember'].mean():.0%}")

    st.subheader("Average characteristics: churned vs retained")
    averages = (
        customers.groupby(TARGET)[["Age", "Balance", "NumOfProducts", "Tenure", "CreditScore", "IsActiveMember"]]
        .mean()
        .rename(index={0: "Retained", 1: "Churned"})
        .rename_axis(None)
        .rename(columns={"NumOfProducts": "Products", "CreditScore": "Credit score",
                         "IsActiveMember": "Share active"})
    )
    st.dataframe(
        averages.style.format({"Age": "{:.1f}", "Balance": "{:,.0f}", "Products": "{:.2f}",
                               "Tenure": "{:.1f}", "Credit score": "{:.0f}", "Share active": "{:.0%}"}),
        width="stretch",
    )

    st.subheader("Churn rate by segment")
    st.caption(f"Dashed line = overall churn rate ({overall_churn:.1f}%). Hover over bars for customer counts.")
    left, right = st.columns(2)
    left.altair_chart(churn_bar_chart(customers, "Geography", "By country", overall_rate=overall_churn),
                      width="stretch")
    right.altair_chart(churn_bar_chart(customers, "AgeBand", "By age band", sort=AGE_BANDS,
                                       overall_rate=overall_churn), width="stretch")
    left.altair_chart(churn_bar_chart(customers, "NumOfProducts", "By number of products",
                                      overall_rate=overall_churn), width="stretch")
    right.altair_chart(churn_bar_chart(customers, "Activity", "By activity status",
                                       overall_rate=overall_churn), width="stretch")

    st.subheader("Key findings")
    st.markdown(
        """
- **About 1 in 5 customers (20.4%) churned.**
- **Age** shows the strongest pattern: churn peaks at **56% for customers aged 50–59**, versus under 11% for under-40s.
- **Number of products** is non-linear: **2 products → 7.6%** churn (lowest), 1 product → 27.7%, 3–4 products → 83–100% (small groups).
- **Inactive customers churn almost twice as often** as active ones (26.9% vs 14.3%).
- **Germany's churn rate (32.4%) is about double** that of France and Spain (~16%).
- Credit score, estimated salary, tenure and credit-card ownership show **little or no association** with churn.

*These are associations in historical data, not proof of cause and effect.*
"""
    )

# ---------------------------------------------------------------------------
# Page 2: Customer analysis with filters
# ---------------------------------------------------------------------------
elif page == "Customer analysis":
    st.title("Customer Segment Analysis")
    st.caption("Filter the customer base and see how churn changes across segments.")

    f1, f2, f3 = st.columns(3)
    geographies = f1.multiselect("Country", sorted(customers["Geography"].unique()),
                                 default=sorted(customers["Geography"].unique()))
    genders = f2.multiselect("Gender", sorted(customers["Gender"].unique()),
                             default=sorted(customers["Gender"].unique()))
    activity = f3.selectbox("Active membership", ["All", "Active", "Inactive"])
    f4, f5 = st.columns([2, 1])
    age_range = f4.slider("Age range", int(customers["Age"].min()), int(customers["Age"].max()),
                          (int(customers["Age"].min()), int(customers["Age"].max())))
    products = f5.multiselect("Number of products", [1, 2, 3, 4], default=[1, 2, 3, 4])

    segment = customers[
        customers["Geography"].isin(geographies)
        & customers["Gender"].isin(genders)
        & customers["Age"].between(*age_range)
        & customers["NumOfProducts"].isin(products)
    ]
    if activity != "All":
        segment = segment[segment["Activity"] == activity]

    if segment.empty:
        st.warning("No customers match these filters. Try widening the selection.")
        st.stop()

    segment_churn = segment[TARGET].mean() * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers in segment", f"{len(segment):,}",
              help=f"{len(segment) / len(customers):.0%} of all customers")
    m2.metric("Churned", f"{segment[TARGET].sum():,}")
    difference = segment_churn - overall_churn
    m3.metric("Segment churn rate", f"{segment_churn:.1f}%",
              f"{difference:+.1f} pp vs overall" if abs(difference) >= 0.05 else None,
              delta_color="inverse")
    m4.metric("Average age", f"{segment['Age'].mean():.1f}")

    if len(segment) < 100:
        st.info(f"This segment has only {len(segment)} customers, so churn rates may be unreliable.")

    breakdown_options = {
        "Country": "Geography", "Gender": "Gender", "Age band": "AgeBand",
        "Number of products": "NumOfProducts", "Activity status": "Activity",
        "Credit card": "CreditCard", "Tenure (years)": "Tenure",
    }
    breakdown_label = st.selectbox("Break the segment down by", list(breakdown_options))
    breakdown = breakdown_options[breakdown_label]
    st.altair_chart(
        churn_bar_chart(segment, breakdown, f"Churn rate by {breakdown_label.lower()} (filtered segment)",
                        sort=AGE_BANDS if breakdown == "AgeBand" else None, overall_rate=overall_churn),
        width="stretch",
    )
    st.caption(f"Dashed line = overall churn rate across all customers ({overall_churn:.1f}%).")

    table = churn_rate_table(segment, breakdown).rename(
        columns={breakdown: breakdown_label, "customers": "Customers", "churned": "Churned",
                 "churn_rate": "Churn rate (%)"})
    st.dataframe(table, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Page 3: Churn risk predictor
# ---------------------------------------------------------------------------
elif page == "Churn risk predictor":
    st.title("Churn Risk Predictor")
    if model is None:
        st.error("Model file not found. Run notebooks/03_model_comparison.ipynb to create models/churn_model.joblib.")
        st.stop()

    st.caption(f"Enter a customer's details to estimate their churn probability with the "
               f"{metadata['model_name']} model.")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        geography = c1.selectbox("Country", ["France", "Germany", "Spain"])
        gender = c1.selectbox("Gender", ["Female", "Male"])
        age = c1.number_input("Age", min_value=18, max_value=92, value=40)
        tenure = c2.slider("Tenure (years with the bank)", 0, 10, 5)
        num_products = c2.selectbox("Number of products", [1, 2, 3, 4], index=0)
        is_active = c2.radio("Active member?", ["Yes", "No"], horizontal=True)
        has_card = c2.radio("Has a credit card?", ["Yes", "No"], horizontal=True)
        balance = c3.number_input("Account balance", min_value=0.0, max_value=300_000.0,
                                  value=100_000.0, step=5_000.0)
        credit_score = c3.slider("Credit score", 350, 850, 650)
        salary = c3.number_input("Estimated salary", min_value=0.0, max_value=250_000.0,
                                 value=100_000.0, step=5_000.0)
        submitted = st.form_submit_button("Estimate churn risk", type="primary")

    if submitted:
        customer = pd.DataFrame([{
            "Geography": geography, "Gender": gender, "HasCrCard": int(has_card == "Yes"),
            "IsActiveMember": int(is_active == "Yes"), "CreditScore": credit_score, "Age": age,
            "Tenure": tenure, "Balance": balance, "NumOfProducts": num_products,
            "EstimatedSalary": salary,
        }])[FEATURES]
        probability = float(model.predict_proba(customer)[0, 1])
        tier = str(assign_risk_tier([probability])[0])

        r1, r2 = st.columns(2)
        r1.metric("Predicted churn probability", f"{probability:.0%}")
        r2.metric("Risk level", RISK_BADGES[tier])
        st.progress(probability)

        # Suggested action from the risk × value framework in notebook 05
        higher_risk = probability >= metadata["decision_threshold"]
        higher_value = balance >= load_balance_cutoff()
        actions = {
            (True, True): ("Priority outreach", "Relationship-manager contact and a tailored retention offer."),
            (True, False): ("Automated engagement", "Low-cost digital engagement (personalised nudges, product tips)."),
            (False, True): ("Nurture", "Maintain service quality and monitor for changes in behaviour."),
            (False, False): ("Standard", "No proactive action needed; standard service."),
        }
        action, description = actions[(higher_risk, higher_value)]
        st.markdown(f"**Suggested retention approach: {action}.** {description}")

        tier_text = ", ".join(f"{name} = {low:.0%}–{min(high, 1):.0%}" for name, low, high in RISK_TIERS)
        st.info(
            "**This is a statistical estimate, not a guarantee.** The model estimates how often customers "
            "with a similar profile left the bank in historical data. It doesn't know this customer's "
            "personal circumstances and can't say *why* they might leave. "
            f"Risk tiers: {tier_text}. 'Higher risk' for the suggested approach means probability ≥ "
            f"{metadata['decision_threshold']:.0%}, and 'higher value' means balance ≥ the median balance "
            "(a rough proxy: the dataset has no profitability data)."
        )

# ---------------------------------------------------------------------------
# Page 4: Model performance
# ---------------------------------------------------------------------------
else:
    st.title("Model Performance")
    st.caption("All models evaluated on the same 2,000 held-out test customers (20.3% churned).")

    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics.style.format({c: "{:.3f}" for c in metrics.columns if c != "Model"}),
                 hide_index=True, width="stretch")
    st.caption("Metrics at the default 0.5 probability threshold. Precision, recall and F1 refer to the churn class.")

    if metadata:
        default = metadata["test_metrics_default_threshold"]
        business = metadata["test_metrics_business_threshold"]
        st.subheader(f"Selected model: {metadata['model_name']}")
        st.markdown(
            f"""
- Chosen by **cross-validated ROC-AUC ({metadata['cv_roc_auc']:.3f})** on the training data, before looking at the test set.
- At the default 0.5 threshold it catches **{default['Recall']:.0%} of churners** (precision {default['Precision']:.0%}).
- Lowering the threshold to **{metadata['decision_threshold']}** (target: catch ≥ 70% of churners) raises recall to
  **{business['Recall']:.0%}**, with precision **{business['Precision']:.0%}**: more churners caught, but more false alarms.
- **Why not accuracy?** A model that predicts "nobody churns" is already ~80% accurate while catching zero churners.
"""
        )

    for figure, caption in [
        ("04_model_comparison_auc.png", "Test ROC-AUC by model (0.5 = random guessing)"),
        ("05_feature_importance.png", "Which features the selected model relies on"),
    ]:
        if (FIGURES_DIR / figure).exists():
            st.image(str(FIGURES_DIR / figure), caption=caption, width=900)
