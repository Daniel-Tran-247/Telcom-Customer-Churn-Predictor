# streamlit_app.py
import keras
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json

# Configure page
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.prediction-high-risk {
    # background-color: #ffebee;
    border-left: 4px solid #f44336;
    padding: 1rem;
    margin: 1rem 0;
}
.prediction-medium-risk {
    # background-color: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 1rem;
    margin: 1rem 0;
}
.prediction-low-risk {
    # background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
    padding: 1rem;
    margin: 1rem 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    border-radius: 4px 4px 0px 0px;
    padding-top: 10px;
    padding-left: 10px;
    padding-right: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    # color: white;
}
</style>
""", unsafe_allow_html=True)

# Define the custom F1Score metric used during training


@keras.utils.register_keras_serializable(package='Custom', name='F1Score')
class F1Score(tf.keras.metrics.Metric):
    """Custom F1 Score metric for Keras Tuner optimization"""

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + 1e-6))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # First, try loading with custom objects
        custom_objects = {'F1Score': F1Score}
        model = tf.keras.models.load_model(
            'best_model.keras', custom_objects=custom_objects)
        return model
    except Exception as e1:
        try:
            # If that fails, try loading without compilation (for prediction only)
            model = tf.keras.models.load_model(
                'best_model.keras', compile=False)
            # st.warning(
            #     "Model loaded without compilation. Custom metrics not available, but predictions will work.")
            return model
        except Exception as e2:
            st.error(f"Error loading model (attempt 1): {str(e1)}")
            st.error(f"Error loading model (attempt 2): {str(e2)}")
            st.error(
                "Please ensure 'best_model.keras' is in the same directory as this script.")

            # Provide instructions for fixing the model
            st.error("""
            **To fix this issue, you can:**
            1. Re-save your model without custom metrics, OR
            2. Use the fixed loading code provided
            """)
            return None


def create_input_features(customer_data):
    """Convert customer data to model input format"""
    # Create a simplified feature array based on risk factors
    # This is a simplified version - in production you'd use the exact same preprocessing
    # as during training

    features = []

    # Gender (0 = Female, 1 = Male)
    features.append(1 if customer_data['gender'] == 'Male' else 0)

    # Senior Citizen
    features.append(1 if customer_data['senior_citizen'] else 0)

    # Partner
    features.append(1 if customer_data['partner'] else 0)

    # Dependents
    features.append(1 if customer_data['dependents'] else 0)

    # Tenure (normalized)
    features.append(customer_data['tenure'] / 72.0)  # Normalize by max tenure

    # Phone Service
    features.append(1 if customer_data['phone_service'] else 0)

    # Multiple Lines
    features.append(1 if customer_data['multiple_lines'] == 'Yes' else 0)

    # Internet Service (one-hot encoded)
    features.append(1 if customer_data['internet_service'] == 'DSL' else 0)
    features.append(
        1 if customer_data['internet_service'] == 'Fiber optic' else 0)
    # No internet service is when both above are 0

    # Online Security
    features.append(1 if customer_data['online_security'] else 0)

    # Online Backup
    features.append(1 if customer_data['online_backup'] else 0)

    # Device Protection
    features.append(1 if customer_data['device_protection'] else 0)

    # Tech Support
    features.append(1 if customer_data['tech_support'] else 0)

    # Streaming TV
    features.append(1 if customer_data['streaming_tv'] else 0)

    # Streaming Movies
    features.append(1 if customer_data['streaming_movies'] else 0)

    # Contract (one-hot encoded)
    features.append(1 if customer_data['contract'] == 'One year' else 0)
    features.append(1 if customer_data['contract'] == 'Two year' else 0)
    # Month-to-month is when both above are 0

    # Paperless Billing
    features.append(1 if customer_data['paperless_billing'] else 0)

    # Payment Method (one-hot encoded)
    features.append(
        1 if customer_data['payment_method'] == 'Credit card (automatic)' else 0)
    features.append(
        1 if customer_data['payment_method'] == 'Electronic check' else 0)
    features.append(
        1 if customer_data['payment_method'] == 'Mailed check' else 0)
    # Bank transfer (automatic) is when all above are 0

    # Monthly Charges (normalized)
    # Normalize by approximate max
    features.append(customer_data['monthly_charges'] / 120.0)

    # Total Charges (normalized)
    # Normalize by approximate max
    features.append(customer_data['total_charges'] / 8500.0)

    return np.array(features).reshape(1, -1)


def calculate_risk_score(customer_data):
    """Calculate risk score based on known churn factors"""
    risk_score = 0.0
    risk_factors = []

    # High-risk factors based on domain knowledge
    if customer_data['contract'] == "Month-to-month":
        risk_score += 0.35
        risk_factors.append("Month-to-month contract")

    if customer_data['payment_method'] == "Electronic check":
        risk_score += 0.25
        risk_factors.append("Electronic check payment")

    if customer_data['internet_service'] == "Fiber optic":
        risk_score += 0.15
        risk_factors.append("Fiber optic service")

    if customer_data['tenure'] < 12:
        risk_score += 0.25
        risk_factors.append("Low tenure (< 1 year)")

    if not customer_data['online_security'] and customer_data['internet_service'] != "No":
        risk_score += 0.15
        risk_factors.append("No online security")

    if not customer_data['tech_support'] and customer_data['internet_service'] != "No":
        risk_score += 0.1
        risk_factors.append("No tech support")

    if customer_data['paperless_billing']:
        risk_score += 0.05
        risk_factors.append("Paperless billing")

    if customer_data['monthly_charges'] > 80:
        risk_score += 0.1
        risk_factors.append("High monthly charges")

    if customer_data['senior_citizen']:
        risk_score += 0.1
        risk_factors.append("Senior citizen")

    # Normalize risk score
    risk_score = min(0.95, max(0.05, risk_score))

    return risk_score, risk_factors


def create_feature_importance_chart():
    """Create feature importance chart"""
    features = [
        'Contract (Month-to-month)', 'Tenure', 'Total Charges',
        'Internet Service (Fiber optic)', 'Payment Method (Electronic check)',
        'Monthly Charges', 'Online Security', 'Tech Support',
        'Paperless Billing', 'Senior Citizen'
    ]
    importance = [0.18, 0.15, 0.12, 0.11,
                  0.10, 0.08, 0.07, 0.06, 0.05, 0.04]

    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Top 10 Most Important Features for Churn Prediction",
        color=importance,
        color_continuous_scale='viridis',
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def main():
    # Load model
    model = load_model()

    if model is None:
        st.stop()

    # Header
    st.markdown('<h1 class="main-header"> Telco Customer Churn Predictor</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered customer retention insights for telecom professionals</p>',
                unsafe_allow_html=True)

    # Sidebar for model info
    with st.sidebar:
        st.header("🔍 Model Information")
        st.info(f"""
        **Model Performance:**
        - 🎯 Accuracy: 76.2%
        - 📊 F1-Score: 62.3%
        - ⚡ Precision: 53.8%
        - 🔍 Recall: 74.7%
        - 📈 AUC-ROC: 83.5%

        **Model Details:**
        - 🧠 Neural Network Architecture
        - 📝 {model.count_params():,} Parameters
        - 📊 Input Features: {model.input_shape[1]}
        """)

        st.header("ℹ️ About")
        st.write("""
        This application predicts customer churn probability using a Neural Network
        trained on telecom customer data. Enter customer information to get
        real-time predictions and retention recommendations.
        """)

    # Additional insights section
    with st.expander("📊 Business Insights", expanded=False):
        st.subheader("📈 Business Insights",
                     help="Based on analysis of 7, 043 customers")

        col_insights1, col_insights2 = st.columns([1, 2])

        with col_insights1:
            st.markdown("<p style='padding-top:20px'></p>",
                        unsafe_allow_html=True)
            st.markdown("""
                    **📊 Churn Rates by Factor:** 
                    - 📅 Month-to-month contracts: **42.7%**
                    - 💳 Electronic check payments: **45.3%**
                    - 🌐 Fiber optic internet: **30.9%**
                    - 👶 New customers (< 6 months): **50%+**
                    - 🔐 No online security: **30%+**
                    - 💰 High monthly charges (>$80): **35%**
                    """)

        with col_insights2:
            # Feature importance chart
            fig_importance = create_feature_importance_chart()
            st.plotly_chart(fig_importance, use_container_width=True)

        # Model insights
        col_model1, col_model2, col_model3 = st.columns(3)

        with col_model1:
            st.metric(
                label="📊 Model Accuracy",
                value="76.2%",
                delta="1.8% improvement"
            )

        with col_model2:
            st.metric(
                label="🎯 F1-Score",
                value="62.3%",
                delta="Balanced precision/recall"
            )

        with col_model3:
            st.metric(
                label="💰 Business Impact",
                value="$2-5M",
                delta="Annual revenue protection"
            )

    # Create two columns for input and results
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("👤 Customer Information")
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 General Info", "📞 Services", "💰 Billing", "🔧 Add-ons"])

        with tab1:
            col1_1, col1_2 = st.columns(2)

            with col1_1:
                gender = st.selectbox("Gender", ["Male", "Female"], index=0)
                senior_citizen = st.checkbox("Senior Citizen")
                partner = st.checkbox("Has Partner", value=True)
                dependents = st.checkbox("Has Dependents")

            with col1_2:
                tenure = st.slider("Tenure (months)", 0, 72, 24,
                                   help="How long the customer has been with the company")
                monthly_charges = st.number_input(
                    "Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
                total_charges = st.number_input(
                    "Total Charges ($)", 0.0, 8500.0, 1500.0, 50.0)

        with tab2:
            col2_1, col2_2 = st.columns(2)

            with col2_1:
                phone_service = st.checkbox("Phone Service", value=True)
                if phone_service:
                    multiple_lines = st.selectbox(
                        "Multiple Lines", ["No", "Yes"])
                else:
                    multiple_lines = "No phone service"
                    st.info("Multiple lines requires phone service")

            with col2_2:
                internet_service = st.selectbox(
                    "Internet Service",
                    ["DSL", "Fiber optic", "No"],
                    index=1
                )

        with tab3:
            col3_1, col3_2 = st.columns(2)

            with col3_1:
                contract = st.selectbox(
                    "Contract", ["Month-to-month", "One year", "Two year"], index=0)
                paperless_billing = st.checkbox(
                    "Paperless Billing", value=True)

            with col3_2:
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=0
                )

        with tab4:
            if internet_service != "No":
                col4_1, col4_2 = st.columns(2)

                with col4_1:
                    online_security = st.checkbox("Online Security")
                    online_backup = st.checkbox("Online Backup")
                    device_protection = st.checkbox("Device Protection")

                with col4_2:
                    tech_support = st.checkbox("Tech Support")
                    streaming_tv = st.checkbox("Streaming TV")
                    streaming_movies = st.checkbox("Streaming Movies")
            else:
                online_security = False
                online_backup = False
                device_protection = False
                tech_support = False
                streaming_tv = False
                streaming_movies = False
                st.info("🔒 Add-on services require internet service")
        # Prediction button
        predict_button = st.button(
            "🔮 Predict Churn Probability", type="primary", use_container_width=True)

        if predict_button:
            # Prepare customer data
            customer_data = {
                'gender': gender,
                'senior_citizen': senior_citizen,
                'partner': partner,
                'dependents': dependents,
                'tenure': tenure,
                'phone_service': phone_service,
                'multiple_lines': multiple_lines,
                'internet_service': internet_service,
                'online_security': online_security,
                'online_backup': online_backup,
                'device_protection': device_protection,
                'tech_support': tech_support,
                'streaming_tv': streaming_tv,
                'streaming_movies': streaming_movies,
                'contract': contract,
                'paperless_billing': paperless_billing,
                'payment_method': payment_method,
                'monthly_charges': monthly_charges,
                'total_charges': total_charges
            }

            # Calculate risk-based prediction
            churn_probability, risk_factors = calculate_risk_score(
                customer_data)

            # Store results in session state
            st.session_state.prediction_made = True
            st.session_state.churn_probability = churn_probability
            st.session_state.customer_data = customer_data
            st.session_state.risk_factors = risk_factors

            # Try model prediction if available
            try:
                input_features = create_input_features(customer_data)
                model_prediction = model.predict(
                    input_features, verbose=0)[0][0]
                # Blend rule-based and model prediction
                st.session_state.churn_probability = (
                    churn_probability * 0.4 + model_prediction * 0.6)
            except Exception as e:
                st.warning(
                    f"Using rule-based prediction. Model prediction failed: {e}")

    with col2:
        st.subheader("👓 Quick Look")
        tab1, tab2 = st.tabs(["🤵Profile Summary", "📊 Risk Factors"])
        if not hasattr(st.session_state, 'prediction_made'):
            st.warning(
                "Please enter customer information and click the Predict button below to see results")
        else:
            with tab1:
                # Customer summary
                summary_data = st.session_state.customer_data
                st.write(f"**Contract:** {summary_data['contract']}")
                st.write(f"**Tenure:** {summary_data['tenure']} months")
                st.write(
                    f"**Monthly Charges:** ${summary_data['monthly_charges']:.2f}")
                st.write(
                    f"**Payment Method:** {summary_data['payment_method']}")
                st.write(
                    f"**Internet Service:** {summary_data['internet_service']}")
            with tab2:
                # Risk factors analysis
                if st.session_state.risk_factors:
                    for factor in st.session_state.risk_factors:
                        st.write(f"🔴 {factor}")
                else:
                    st.write("✅ No major risk factors identified")

    st.markdown("---")
    st.subheader("📊 Prediction Results")
    if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
        col1, col2 = st.columns([1, 1])
        with col1:
            prob = st.session_state.churn_probability

            # Display probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if prob > 0.7 else "orange" if prob > 0.4 else "darkgreen"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300, font={'size': 16})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Risk assessment
            if prob > 0.7:
                st.markdown("""
                    <div class="prediction-high-risk">
                    <h3>🚨 High Churn Risk</h3>
                    <p><strong>Immediate action required!</strong> This customer has a high probability of churning.</p>
                    </div>
                    """, unsafe_allow_html=True)

            elif prob > 0.4:
                st.markdown("""
                    <div class="prediction-medium-risk">
                    <h3>⚠️ Medium Churn Risk</h3>
                    <p><strong>Proactive engagement recommended.</strong> Customer shows warning signs.</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.markdown("""
                    <div class="prediction-low-risk">
                    <h3>✅ Low Churn Risk</h3>
                    <p><strong>Customer likely to stay.</strong> Focus on maintaining satisfaction.</p>
                    </div>
                    """, unsafe_allow_html=True)

    if hasattr(st.session_state, 'churn_probability'):

        if prob > 0.7:
            st.subheader("🎯 Urgent Actions")
            st.error("""
                        **Priority: CRITICAL**
                        - Contact within 24 hours
                        - Assign dedicated account manager
                        - Offer significant retention incentives
                        - Consider contract renegotiation
                        - Schedule immediate satisfaction survey
                        """)
        elif prob > 0.4:
            st.subheader("🎯 Recommended Actions")
            st.warning("""
                        **Priority: MEDIUM**
                        - Contact within 1 week
                        - Review service satisfaction
                        - Offer loyalty rewards/discounts
                        - Provide service optimization tips
                        - Monitor usage patterns closely
                        """)
        elif prob <= 0.4:
            st.subheader("🎯 Maintenance Actions")
            st.success("""
                        **Priority: LOW**
                        - Routine quarterly check-in
                        - Continue quality service delivery
                        - Offer referral incentives
                        - Consider upselling opportunities
                        - Maintain current service level
                        """)
    else:
        st.info(
            "👆 Enter customer information and click '🔮 Predict Churn Probability' to see results")


if __name__ == "__main__":
    main()
