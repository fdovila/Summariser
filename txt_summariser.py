# Source: https://huggingface.co/human-centered-summarization/financial-summarization-pegasus
# You need Python 3.8.5 with the following packages via PIP: torch, tensorflow, and sentencepiece
# It takes ~1m for 300 words texts on CPU (i5). GPU should run much faster

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("human-centered-summarization/financial-summarization-pegasus")

model = AutoModelForSeq2SeqLM.from_pretrained("human-centered-summarization/financial-summarization-pegasus")

# Some text to summarize here
text_to_summarize = "INSERT YOUR LONG TEXT HERE"

# Tokenize our text
# If you want to run the code in Tensorflow, please remember to return the particular tensors as simply as using return_tensors = 'tf'
input_ids = tokenizer(text_to_summarize, return_tensors="pt").input_ids

# Generate the output (Here, we use beam search but you can also use any other strategy you like)
output = model.generate(
    input_ids,
    max_length=32,
    num_beams=5,
    early_stopping=True
)

# Finally, we can print the generated summary
print(tokenizer.decode(output[0], skip_special_tokens=True))
