from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

checkpoint = "vesteinn/IceBERT"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


dataset = load_dataset("Sigurdur/imdb-isl-google-translate")

def preprocess_function(examples):
    return tokenizer(examples["review"], truncation=True)

