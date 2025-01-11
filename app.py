from transformers import pipeline
import pandas as pd
import streamlit as st
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Load the summarization pipeline
summarizer = pipeline("summarization", model="t5-small")  # Or use "facebook/bart-large-cnn"

st.title("Text Summarizer")
input_text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    summary = summarizer(input_text, max_length=50, min_length=25, do_sample=False)
    st.subheader("Summary")
    st.write(summary[0]['summary_text'])