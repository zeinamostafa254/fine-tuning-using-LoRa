import sys
from pathlib import Path

import streamlit as st


 
# PATH SETUP
 

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


 
# IMPORT MODEL FUNCTIONS
 

from utils import (
    load_model,
    load_tokenizer,
    predict_sentiment
)

from preprocessing import clean_text


 
# PAGE CONFIGURATION
 

st.set_page_config(
    page_title="IMDB Sentiment AI",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)


 
# CUSTOM CSS
 

st.html(
    """
    <style>

    /*  
       MAIN APPLICATION
         */

    .stApp {

        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(139, 92, 246, 0.18),
                transparent 30%
            ),

            radial-gradient(
                circle at 90% 15%,
                rgba(236, 72, 153, 0.15),
                transparent 30%
            ),

            linear-gradient(
                135deg,
                #0f1020 0%,
                #17152d 50%,
                #111827 100%
            );

        color: #f8fafc;
    }


    /*  
       HIDE STREAMLIT DEFAULT ELEMENTS
         */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /*  
       MAIN CONTAINER
         */

    .block-container {

        max-width: 850px;

        padding-top: 3rem;
        padding-bottom: 3rem;
    }


    /*  
       HERO SECTION
         */

    .hero {

        text-align: center;

        padding:
            1rem 0 2.2rem 0;
    }


    .hero-icon {

        font-size: 4rem;

        line-height: 1;

        margin-bottom: 1rem;
    }


    .hero-title {

        font-size: 3rem;

        font-weight: 800;

        line-height: 1.1;

        background:
            linear-gradient(
                90deg,
                #a78bfa,
                #f472b6,
                #60a5fa
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        background-clip: text;
    }


    .hero-subtitle {

        color: #cbd5e1;

        font-size: 1.05rem;

        margin-top: 0.8rem;
    }


    .hero-model {

        color: #94a3b8;

        font-size: 0.85rem;

        margin-top: 0.5rem;
    }


    /*  
       INPUT CARD
         */

    .glass-card {

        background:
            rgba(255, 255, 255, 0.055);

        border:
            1px solid
            rgba(255, 255, 255, 0.10);

        border-radius: 22px;

        padding: 1.5rem;

        margin: 1rem 0;

        box-shadow:
            0 15px 45px
            rgba(0, 0, 0, 0.20);

        backdrop-filter:
            blur(12px);
    }


    /*  
       SECTION TITLE
         */

    .section-title {

        color: #f8fafc;

        font-size: 1.15rem;

        font-weight: 700;

        margin-bottom: 0.8rem;
    }


    /*  
       TEXT INPUT
         */

    textarea {

        background: #ffffff !important;

        color: #111827 !important;

        border:
            1px solid
            rgba(167, 139, 250, 0.35) !important;

        border-radius: 16px !important;

        font-size: 1rem !important;

        line-height: 1.6 !important;
    }


    /* Text entered by the user */

    textarea:focus {

        background: #ffffff !important;

        color: #111827 !important;

        border:
            1px solid
            #8b5cf6 !important;

        box-shadow:
            0 0 0 2px
            rgba(139, 92, 246, 0.15) !important;
    }


    /* Placeholder */

    textarea::placeholder {

        color: #6b7280 !important;

        opacity: 1 !important;
    }


    /*  
       ANALYZE BUTTON
         */

    .stButton > button {

        width: 100%;

        border: none;

        border-radius: 14px;

        padding:
            0.75rem 1rem;

        font-size: 1rem;

        font-weight: 700;

        color: white;

        background:
            linear-gradient(
                90deg,
                #7c3aed,
                #db2777
            );

        transition:
            all 0.25s ease;

        box-shadow:
            0 8px 25px
            rgba(124, 58, 237, 0.25);
    }


    .stButton > button:hover {

        transform:
            translateY(-2px);

        box-shadow:
            0 12px 30px
            rgba(219, 39, 119, 0.30);
    }


    /*  
       RESULT CARDS
         */

    .positive-result {

        background:
            linear-gradient(
                135deg,
                rgba(34, 197, 94, 0.16),
                rgba(16, 185, 129, 0.08)
            );

        border:
            1px solid
            rgba(34, 197, 94, 0.30);

        border-radius: 20px;

        padding: 1.5rem;

        text-align: center;

        margin-top: 1rem;
    }


    .negative-result {

        background:
            linear-gradient(
                135deg,
                rgba(239, 68, 68, 0.16),
                rgba(244, 63, 94, 0.08)
            );

        border:
            1px solid
            rgba(239, 68, 68, 0.30);

        border-radius: 20px;

        padding: 1.5rem;

        text-align: center;

        margin-top: 1rem;
    }


    .result-emoji {

        font-size: 2.8rem;

        line-height: 1.2;

        margin-bottom: 0.4rem;
    }


    .positive-title {

        color: #4ade80;

        font-size: 2rem;

        font-weight: 800;
    }


    .negative-title {

        color: #fb7185;

        font-size: 2rem;

        font-weight: 800;
    }


    .confidence {

        color: #cbd5e1;

        font-size: 0.95rem;

        margin-top: 0.5rem;
    }


    /*  
       MODEL INFORMATION CARDS
         */

    .stats-container {

        display: flex;

        gap: 12px;

        margin-top: 1rem;
    }


    .stat-card {

        flex: 1;

        background:
            rgba(255, 255, 255, 0.045);

        border:
            1px solid
            rgba(255, 255, 255, 0.08);

        border-radius: 16px;

        padding: 1rem;

        text-align: center;
    }


    .stat-number {

        font-size: 1.2rem;

        font-weight: 800;

        color: #c4b5fd;
    }


    .stat-label {

        font-size: 0.78rem;

        color: #94a3b8;

        margin-top: 0.3rem;
    }


    /*  
       FOOTER
         */

    .footer {

        text-align: center;

        color: #64748b;

        font-size: 0.8rem;

        margin-top: 2rem;

        padding-bottom: 1rem;
    }


    /*  
       MOBILE RESPONSIVENESS
         */

    @media (max-width: 600px) {

        .hero-title {

            font-size: 2.3rem;
        }

        .stats-container {

            flex-direction: column;
        }

    }

    </style>
    """
)


 
# HERO SECTION
 

st.html(
    """
    <div class="hero">

        <div class="hero-icon">
            🎬
        </div>

        <div class="hero-title">
            IMDB Sentiment AI
        </div>

        <div class="hero-subtitle">
            Let AI determine whether a movie review
            is positive or negative.
        </div>

        <div class="hero-model">
            Powered by DistilBERT + LoRA
        </div>

    </div>
    """
)


 
# LOAD MODEL
 

@st.cache_resource
def initialize_model():

    tokenizer = load_tokenizer()

    model = load_model()

    return model, tokenizer


with st.spinner("✨ Loading the AI model..."):

    try:

        model, tokenizer = initialize_model()

    except Exception as error:

        st.error(
            "The AI model could not be loaded."
        )

        st.code(
            str(error)
        )

        st.stop()


 
# REVIEW INPUT HEADER
 

st.html(
    """
    <div class="glass-card">

        <div class="section-title">
            ✍️ Enter a movie review
        </div>

    </div>
    """
)


 
# REVIEW TEXT AREA
 

review = st.text_area(
    "Movie review",
    height=180,
    placeholder="Write your movie review here...",
    label_visibility="collapsed"
)


 
# CHARACTER COUNT
 

character_count = len(review)

st.caption(
    f"{character_count} characters"
)


 
# ANALYZE BUTTON
 

analyze = st.button(
    "✨ Analyze Sentiment"
)


 
# SENTIMENT PREDICTION
 

if analyze:

     
    # Empty input
     

    if not review.strip():

        st.warning(
            "Please enter a movie review first. 🍿"
        )


      
    # Valid input
      

    else:

        # Basic preprocessing
        cleaned_review = clean_text(
            review
        )

        # Model inference
        with st.spinner(
            "🔮 Analyzing your review..."
        ):

            result = predict_sentiment(
                cleaned_review,
                model,
                tokenizer
            )


         
        # Get prediction information
         

        sentiment = result["sentiment"]

        confidence = result["confidence"]

        negative_probability = (
            result["negative_probability"]
        )

        positive_probability = (
            result["positive_probability"]
        )


         
        # POSITIVE RESULT
         

        if sentiment == "POSITIVE":

            st.html(
                f"""
                <div class="positive-result">

                    <div class="result-emoji">
                        😊
                    </div>

                    <div class="positive-title">
                        POSITIVE
                    </div>

                    <div class="confidence">

                        The model is

                        <strong>
                            {confidence * 100:.1f}%
                        </strong>

                        confident that this review
                        is positive.

                    </div>

                </div>
                """
            )


         
        # NEGATIVE RESULT
         

        else:

            st.html(
                f"""
                <div class="negative-result">

                    <div class="result-emoji">
                        😞
                    </div>

                    <div class="negative-title">
                        NEGATIVE
                    </div>

                    <div class="confidence">

                        The model is

                        <strong>
                            {confidence * 100:.1f}%
                        </strong>

                        confident that this review
                        is negative.

                    </div>

                </div>
                """
            )


         
        # PROBABILITIES
         

        st.markdown(
            "### 🎯 Prediction probabilities"
        )


        # Negative probability

        st.write(
            f"Negative — "
            f"{negative_probability * 100:.1f}%"
        )

        st.progress(
            negative_probability
        )


        # Positive probability

        st.write(
            f"Positive — "
            f"{positive_probability * 100:.1f}%"
        )

        st.progress(
            positive_probability
        )


 
# MODEL INFORMATION
 

st.markdown(
    "### 🧠 About the model"
)


st.html(
    """
    <div class="stats-container">

        <div class="stat-card">

            <div class="stat-number">
                DistilBERT
            </div>

            <div class="stat-label">
                Base Model
            </div>

        </div>


        <div class="stat-card">

            <div class="stat-number">
                LoRA
            </div>

            <div class="stat-label">
                Fine-Tuning
            </div>

        </div>


        <div class="stat-card">

            <div class="stat-number">
                2
            </div>

            <div class="stat-label">
                Classes
            </div>

        </div>

    </div>
    """
)


 
# TECHNICAL DETAILS
 

with st.expander(
    "🔬 Technical details"
):

    st.markdown(
        """
        **Base model:** DistilBERT

        **Fine-tuning method:** LoRA

        **Task:** Binary sentiment classification

        **Dataset:** IMDB Movie Reviews

        **LoRA rank:** 8

        **LoRA alpha:** 16

        **LoRA dropout:** 0.1

        **Adapted modules:** `q_lin` and `v_lin`

        **Maximum sequence length:** 256 tokens

        **Number of classes:** 2

        **Labels:**

        - `0` → Negative
        - `1` → Positive

        **Trainable parameters:** 739,586

        **Total parameters:** 67,694,596

        **Trainable percentage:** 1.0925%
        """
    )


 
# FOOTER
 

st.html(
    """
    <div class="footer">

        Built with ❤️ using
        DistilBERT + LoRA + Streamlit

        <br><br>

        IMDB Sentiment Classification Project

    </div>
    """
)