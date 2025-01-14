from datasets import load_dataset 
from transformers import pipeline


#loading the dataset 
xsum_dataset = load_dataset(
    "xsum", 
    version="1.2.0", 
    cache_dir='documents/Huggin_Face/data'
)  # Note: We specify cache_dir to use predownloaded data.
xsum_dataset  
# The printed representation of this object shows the `num_rows` 
# of each dataset split.

xsum_sample = xsum_dataset["train"].select(range(10))

#loading the summarization pipeline
summarizer = pipeline(
    task="summarization",
    model="t5-small",
    min_length=20,
    max_length=40,
    truncation=True,
    model_kwargs={"cache_dir": 'documents/Huggin_Face/'},
)  # Note: We specify cache_dir to use predownloaded models.



# Ask the user for input
input_text = input("Enter the text you want to summarize: ")

# Generate the summary
summary = summarizer(input_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

bullet_points = summary.split(". ")

for point in bullet_points:
    
    print(f"- {point}")

# Print the generated summary
print("Summary:", summary)