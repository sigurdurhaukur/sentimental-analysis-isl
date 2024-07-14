import re
import torch
from utils import Sentiment
from datasets import load_dataset
from transformers import pipeline


def translate_to_is(example):
    text = example["text"]

    # replace all html tags with a space
    text = re.sub(r"<.*?>", " ", text)
    # remove all non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # remove all whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.split()
    batches = [
        " ".join(text[i : i + batch_size]) for i in range(0, len(text), batch_size)
    ]
    translated = pipe(batches, src_lang="en_XX", tgt_lang="is_IS")
    cleaned = " ".join([t["translation_text"] for t in translated])

    example["text"] = cleaned

    return example


if __name__ == "__main__":
    ds = load_dataset("stanfordnlp/imdb")

    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    pipe = pipeline(
        "translation",
        model="mideind/nmt-doc-en-is-2022-10",
        src_lang="en_XX",
        tgt_lang="is_IS",
        device=device,
    )

    batch_size = 128  # n words to translate at a time
    imdb_translated_to_is = ds["train"].map(translate_to_is)  # uses batch size

    sentiment = Sentiment()
    labels = sentiment(imdb_translated_to_is["text"])
    imdb_translated_to_is = imdb_translated_to_is.add_column("emotion", labels)
