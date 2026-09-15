from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


 
# Paths
 

BASE_MODEL_NAME = "distilbert-base-uncased"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"


 
# Device
 

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


 
# Load tokenizer
 

def load_tokenizer():
    """
    Load the tokenizer used during training.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    return tokenizer


 
# Load model
 

def load_model():
    """
    Load the original DistilBERT model and attach
    the trained LoRA adapter.
    """

    # Load the original DistilBERT classification model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        id2label={
            0: "NEGATIVE",
            1: "POSITIVE"
        },
        label2id={
            "NEGATIVE": 0,
            "POSITIVE": 1
        }
    )

    # Load the trained LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        str(MODEL_DIR)
    )

    # Move model to GPU if available
    model = model.to(DEVICE)

    # Evaluation mode
    model.eval()

    return model


 
# Prediction
 

def predict_sentiment(
    text,
    model,
    tokenizer,
    max_length=256
):
    """
    Predict the sentiment of a movie review.

    Returns:
        sentiment
        confidence
        probabilities
    """

    # Tokenize the input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )

    # Move inputs to the same device as the model
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # No gradients are needed during inference
    with torch.no_grad():

        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    # Get predicted class
    predicted_class = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    # Convert class to sentiment
    if predicted_class == 1:
        sentiment = "POSITIVE"
    else:
        sentiment = "NEGATIVE"

    return {
        "sentiment": sentiment,
        "label": predicted_class,
        "confidence": confidence,
        "negative_probability": probabilities[0, 0].item(),
        "positive_probability": probabilities[0, 1].item()
    }