# IMDB Sentiment Analysis with LoRA

A sentiment analysis project that fine-tunes a pre-trained Transformer model on the IMDB movie review dataset using **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)**.

The project classifies movie reviews as either **positive** or **negative** and provides a Streamlit interface for inference.

## Overview

The project demonstrates how LoRA can adapt a pre-trained Transformer for a downstream NLP task while keeping most of the original model parameters frozen.

### Key Concepts

* Transformer-based text classification
* IMDB sentiment analysis
* LoRA fine-tuning
* Parameter-Efficient Fine-Tuning (PEFT)
* Model evaluation
* Streamlit deployment

## Project Structure

```text
fine-tuning-using-LoRa/
│
├── data/          # Dataset files
├── models/        # Fine-tuned model / LoRA adapters
├── notebooks/     # Training and experimentation
├── results/       # Evaluation results
├── src/           # Utility and supporting code
├── app.py         # Streamlit application
├── pyproject.toml # Project dependencies and configuration
└── README.md
```

## LoRA Fine-Tuning

Instead of updating all parameters of the pre-trained model, LoRA introduces small trainable low-rank matrices while keeping the original model weights frozen.

This results in:

* Fewer trainable parameters
* Lower memory requirements
* More efficient fine-tuning
* Smaller task-specific adapters

The number of trainable parameters can be inspected during training using the PEFT parameter report.

## Dataset

The project uses the **IMDB Movie Reviews Dataset** for binary sentiment classification.

| Label | Sentiment |
| ----- | --------- |
| 0     | Negative  |
| 1     | Positive  |

## Installation

Clone the repository:

```bash
git clone https://github.com/zeinamostafa254/fine-tuning-using-LoRa.git
cd fine-tuning-using-LoRa
```

Install the project dependencies:

```bash
pip install -e .
```

## Training

The training and experimentation workflow is available in the `notebooks/` directory.

The main pipeline is:

```text
IMDB Dataset
     ↓
Tokenization
     ↓
Pre-trained Transformer
     ↓
LoRA / PEFT
     ↓
Fine-Tuning
     ↓
Evaluation
     ↓
Saved Adapter
```

## Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application allows users to enter a movie review and receive the predicted sentiment.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face PEFT
* LoRA
* Hugging Face Datasets
* Streamlit
* Scikit-learn

