import pickle
import pandas as pd

# Import ML model
with open('model/model.pkl','rb') as f:
    model = pickle.load(f)

# MLFlow
MODEL_VERSION = '1.0.0'

# Get class labels from model output
class_labels = model.classes_.tolist()

#  Response Model
def predict_output(user_input: dict):
    
    # Input DataFrame
    df = pd.DataFrame([user_input])
    # Predict the class
    predicted_class = model.predict(df)[0]
    # Predict probabilities for all classes
    probabilities = model.predict_proba(df)[0]
    # Confidence score
    confidence = max(probabilities)
    # Create mapping: {class_name: probability}
    class_probs = dict(zip(class_labels, map(lambda p: round(p,4), probabilities)))

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence,4),
        "class_probabilities": class_probs
    }
