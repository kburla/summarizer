import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Input text
text = (
    "Most recently Alice took the stage as Gala Artist of the 40° Festival Internacional de Piano UIS in Bucaramanga, Colombia, performing a sold-out recital as well as a concerto performance with the Orquesta Sinfónica UNAB and Maestro Eduardo Carrizosa. Other highlights include recitals in venues such as Dubai Opera, Carnegie Hall, Teatro la Fenice and the World Economic Forum, as well as in major festivals including DAVOS Festival, Musikdorf Ernen, Verbier Festival Academy and International Sommerakademie Mozarteum. Alice has performed as a soloist with the Brooklyn Philharmonic, Manhattan Chamber Orchestra, Donetsk Philharmonic Orchestra, Torun Symphony Orchestra and Chamber Orchestra Amadeus."
)

# Tokenize input
inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

# Generate summary
summary_ids = model.generate(
    inputs,
    max_length=50,
    min_length=10,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

# Decode summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:", summary)