# backend.py
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Status codes
STATUS_OK = 200
STATUS_BAD_REQUEST = 400
STATUS_INTERNAL_ERROR = 500
STATUS_SERVICE_UNAVAILABLE = 503

# Custom F1Score metric (needed for model loading)


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


# Global variables
model = None
preprocessor = None
feature_names = None


def initialize_backend_model():
    """
    Initialize the model and preprocessor for predictions

    Returns:
        tuple: (status_code, response_json)
    """
    global model, preprocessor, feature_names

    try:
        logger.info("Initializing backend model...")

        # Load the trained model with custom objects
        try:
            custom_objects = {'F1Score': F1Score}
            model = tf.keras.models.load_model(
                'best_model.keras', custom_objects=custom_objects)
            logger.info("Model loaded successfully with custom objects")
        except Exception as e1:
            logger.warning(f"Failed to load with custom objects: {e1}")
            # Try loading without compilation
            try:
                model = tf.keras.models.load_model(
                    'best_model.keras', compile=False)
                logger.info("Model loaded successfully without compilation")
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise e2

        # Initialize preprocessor with the same configuration as training
        preprocessor = create_preprocessor()

        # Define feature names (should match training order)
        feature_names = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]

        response = {
            "status": "success",
            "message": "Model initialized successfully",
            "model_summary": {
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "total_params": int(model.count_params())
            }
        }

        logger.info("Backend initialization completed successfully")
        return STATUS_OK, json.dumps(response)

    except FileNotFoundError as e:
        error_msg = f"Model file not found: {str(e)}"
        logger.error(error_msg)
        response = {
            "status": "error",
            "message": error_msg,
            "error_type": "FileNotFoundError"
        }
        return STATUS_SERVICE_UNAVAILABLE, json.dumps(response)

    except Exception as e:
        error_msg = f"Failed to initialize model: {str(e)}"
        logger.error(error_msg)
        response = {
            "status": "error",
            "message": error_msg,
            "error_type": type(e).__name__
        }
        return STATUS_INTERNAL_ERROR, json.dumps(response)


def create_preprocessor():
    """
    Create the preprocessing pipeline that matches the training configuration

    Returns:
        ColumnTransformer: The preprocessing pipeline
    """
    try:
        # Define feature categories (must match training exactly)
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]

        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first',
             sparse_output=False, handle_unknown='ignore'))
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )

        logger.info("Preprocessor created successfully")
        return preprocessor

    except Exception as e:
        logger.error(f"Error creating preprocessor: {str(e)}")
        raise


def validate_input_data(data):
    """
    Validate the input data structure and values

    Args:
        data (list): List of input values

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check if data is a list
        if not isinstance(data, list):
            return False, "Input data must be a list"

        # For simplified validation, check basic structure
        if len(data) < 19:  # Minimum expected features
            return False, f"Expected at least 19 features, got {len(data)}"

        # Check for numeric values in expected positions
        # tenure, MonthlyCharges, TotalCharges positions
        numeric_positions = [4, 17, 18]
        for pos in numeric_positions:
            if pos < len(data) and not isinstance(data[pos], (int, float)):
                return False, f"Feature at position {pos} must be numeric"

        return True, "Valid input"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def predict_score(input_json):
    """
    Make a prediction using the loaded model

    Args:
        input_json (str): JSON string containing input features

    Returns:
        tuple: (status_code, response_json)
    """
    global model, preprocessor

    try:
        # Check if model is initialized
        if model is None:
            # Try to initialize the model
            status, response = initialize_backend_model()
            if status != STATUS_OK:
                return status, response

        # Parse input JSON
        try:
            input_data = json.loads(input_json)
        except json.JSONDecodeError as e:
            error_response = {
                "status": "error",
                "message": f"Invalid JSON format: {str(e)}",
                "error_type": "JSONDecodeError"
            }
            return STATUS_BAD_REQUEST, json.dumps(error_response)

        # Validate input data
        is_valid, validation_message = validate_input_data(input_data)
        if not is_valid:
            error_response = {
                "status": "error",
                "message": validation_message,
                "error_type": "ValidationError"
            }
            return STATUS_BAD_REQUEST, json.dumps(error_response)

        # Convert input to numpy array and reshape for prediction
        input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)

        # Make prediction
        try:
            prediction_proba = model.predict(input_array, verbose=0)[0][0]
            # Convert to Python float
            prediction_proba = float(prediction_proba)

            # Ensure prediction is between 0 and 1
            prediction_proba = max(0.0, min(1.0, prediction_proba))

            # Binary prediction using 0.5 threshold
            prediction_binary = 1 if prediction_proba > 0.5 else 0

            # Determine risk level
            if prediction_proba > 0.7:
                risk_level = "High"
                risk_description = "Customer likely to churn - immediate action required"
                priority = "URGENT"
            elif prediction_proba > 0.4:
                risk_level = "Medium"
                risk_description = "Customer shows churn indicators - proactive engagement recommended"
                priority = "MEDIUM"
            else:
                risk_level = "Low"
                risk_description = "Customer likely to stay - maintain current service quality"
                priority = "LOW"

            # Calculate confidence score
            confidence = abs(prediction_proba - 0.5) * 2  # How confident (0-1)

            # Generate recommendations
            recommendations = get_recommendation(prediction_proba, input_data)

            # Prepare response
            response = {
                "status": "success",
                "prediction": prediction_proba,
                "binary_prediction": prediction_binary,
                "risk_level": risk_level,
                "risk_description": risk_description,
                "priority": priority,
                "confidence": confidence,
                "recommendation": recommendations,
                "model_version": "1.0.0"
            }

            logger.info(
                f"Prediction successful: {prediction_proba:.4f} (Risk: {risk_level})")
            return STATUS_OK, json.dumps(response)

        except Exception as e:
            error_response = {
                "status": "error",
                "message": f"Model prediction failed: {str(e)}",
                "error_type": "PredictionError"
            }
            logger.error(f"Prediction error: {str(e)}")
            return STATUS_INTERNAL_ERROR, json.dumps(error_response)

    except Exception as e:
        error_response = {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__
        }
        logger.error(f"Unexpected error in predict_score: {str(e)}")
        return STATUS_INTERNAL_ERROR, json.dumps(error_response)


def get_recommendation(churn_probability, input_data):
    """
    Generate specific recommendations based on prediction and input features

    Args:
        churn_probability (float): Predicted churn probability
        input_data (list): Original input features

    Returns:
        list: List of recommendations
    """
    try:
        recommendations = []

        if churn_probability > 0.7:
            recommendations.extend([
                "🚨 URGENT: Contact customer within 24 hours",
                "👨‍💼 Assign dedicated account manager immediately",
                "🎁 Offer significant retention incentives",
                "📋 Consider contract extension with benefits",
                "📞 Schedule immediate satisfaction survey",
                "💰 Review pricing and service bundle options"
            ])

        elif churn_probability > 0.4:
            recommendations.extend([
                "⚡ MEDIUM: Contact customer within 1 week",
                "📊 Review customer satisfaction metrics",
                "🎁 Offer loyalty rewards or discounts",
                "🔧 Provide service optimization recommendations",
                "📈 Monitor usage patterns closely",
                "📞 Schedule proactive check-in call"
            ])

        else:
            recommendations.extend([
                "🌟 LOW: Routine quarterly engagement",
                "✅ Continue current service quality",
                "📈 Consider upselling opportunities",
                "🎁 Offer referral incentives",
                "📞 Maintain regular communication",
                "🔍 Monitor for any service issues"
            ])

        # Add specific recommendations based on risk factors (simplified)
        if len(input_data) > 10:  # Basic check to ensure we have enough data
            try:
                # These would be based on actual feature positions in your model
                # This is a simplified example
                if churn_probability > 0.5:
                    recommendations.append(
                        "📋 Review contract terms and pricing structure")
                    recommendations.append(
                        "🔐 Ensure all security features are enabled")
                    recommendations.append(
                        "💳 Encourage automatic payment methods")
            except:
                pass  # Skip specific recommendations if data parsing fails

        return recommendations

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return ["Standard retention protocol recommended"]


def get_model_info():
    """
    Get information about the loaded model

    Returns:
        tuple: (status_code, response_json)
    """
    global model

    try:
        if model is None:
            # Try to load the model
            status, response = initialize_backend_model()
            if status != STATUS_OK:
                return status, response

        response = {
            "status": "success",
            "model_info": {
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "total_params": int(model.count_params()),
                "model_type": "Sequential Neural Network",
                "framework": "TensorFlow/Keras",
                "version": "1.0.0",
                "expected_features": len(feature_names) if feature_names else "Unknown"
            }
        }

        return STATUS_OK, json.dumps(response)

    except Exception as e:
        error_response = {
            "status": "error",
            "message": f"Error getting model info: {str(e)}",
            "error_type": type(e).__name__
        }
        return STATUS_INTERNAL_ERROR, json.dumps(error_response)


def health_check():
    """
    Health check endpoint for monitoring

    Returns:
        tuple: (status_code, response_json)
    """
    try:
        model_loaded = model is not None

        response = {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "timestamp": pd.Timestamp.now().isoformat(),
            "version": "1.0.0"
        }

        status_code = STATUS_OK if model_loaded else STATUS_SERVICE_UNAVAILABLE
        return status_code, json.dumps(response)

    except Exception as e:
        error_response = {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "error_type": type(e).__name__
        }
        return STATUS_INTERNAL_ERROR, json.dumps(error_response)

# Test function for development


def test_backend():
    """Test function to verify backend functionality"""
    print("🧪 Testing backend functionality...")
    print("=" * 50)

    # Test initialization
    print("1. Testing model initialization...")
    status, response = initialize_backend_model()
    print(f"   Status: {status}")
    response_data = json.loads(response)
    print(f"   Message: {response_data.get('message', 'No message')}")

    if status == STATUS_OK:
        print("   ✅ Model initialization successful")

        # Test health check
        print("\n2. Testing health check...")
        status, response = health_check()
        print(f"   Status: {status}")
        print(f"   Response: {json.loads(response)}")

        # Test model info
        print("\n3. Testing model info...")
        status, response = get_model_info()
        model_info = json.loads(response)
        if status == STATUS_OK:
            print(f"   ✅ Model info retrieved")
            print(f"   Input shape: {model_info['model_info']['input_shape']}")
            print(
                f"   Parameters: {model_info['model_info']['total_params']:,}")

        # Test prediction with sample data
        print("\n4. Testing prediction...")
        # Sample data: simplified feature array
        sample_data = [
            1, 0, 1, 0, 30,  # gender, senior, partner, dependents, tenure
            1, 0, 1, 0, 0,   # phone, multiple, internet_dsl, internet_fiber, online_security
            0, 0, 0, 0, 1,   # backup, device, tech, streaming_tv, streaming_movies
            1, 0, 1, 0, 0,   # contract_one, contract_two, paperless, payment_credit, payment_electronic
            65.0, 2000.0     # monthly_charges, total_charges
        ]

        sample_json = json.dumps(sample_data)

        status, response = predict_score(sample_json)
        print(f"   Status: {status}")

        if status == STATUS_OK:
            pred_data = json.loads(response)
            print(f"   ✅ Prediction successful")
            print(f"   Churn Probability: {pred_data['prediction']:.3f}")
            print(f"   Risk Level: {pred_data['risk_level']}")
            print(f"   Priority: {pred_data['priority']}")
            print(
                f"   Recommendations: {len(pred_data['recommendation'])} items")
        else:
            error_data = json.loads(response)
            print(f"   ❌ Prediction failed: {error_data['message']}")

    else:
        print("   ❌ Model initialization failed")
        error_data = json.loads(response)
        print(f"   Error: {error_data.get('message', 'Unknown error')}")

    print("\n" + "=" * 50)
    print("🏁 Backend testing completed!")


if __name__ == "__main__":
    test_backend()
