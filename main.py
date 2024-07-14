import re
import torch
from tqdm import tqdm
from utils import Sentiment
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


def preprocess_text(txt):
    if isinstance(txt, str):
        txt = [txt]

    # replace all html tags with a space
    texts = [re.sub(r"<.*?>", " ", text) for text in txt]
    # remove all non-alphanumeric characters
    texts = [re.sub(r"[^a-zA-Z0-9\s]", "", text) for text in texts]

    # remove all whitespace
    texts = [re.sub(r"\s+", " ", text) for text in texts]

    return texts


def translate_to_is(
    pipe,
    examples,
    batch_size=128,
    sentiment=None,
    max_seq_length=1024,
    max_char_repitition=4,
):
    # truncate text to max_seq_length
    examples["text"] = [text[:max_seq_length] for text in examples["text"]]

    texts = preprocess_text(examples["text"])

    translated = pipe(
        texts,
        src_lang="en_XX",
        tgt_lang="is_IS",
        batch_size=batch_size,
    )

    examples["text"] = [t["translation_text"] for t in translated]

    # remove chracters repeated more than max_char_repitition times
    examples["text"] = [
        re.sub(r"(.)\1{%d,}" % (max_char_repitition - 1), r"\1", text)
        for text in examples["text"]
    ]

    # add sentiment analysis
    examples["emotions"] = sentiment(examples["text"])

    return examples


def add_sentiment(examples, sentiment):
    examples["emotions"] = sentiment(examples["text"])
    return examples


def sequential_translate_to_is(pipe, example, batch_size, max_char_repitition=4):
    text = example["text"]

    text = preprocess_text(text)[0]
    text = text.split(" ")
    batches = [
        " ".join(text[i : i + batch_size]) for i in range(0, len(text), batch_size)
    ]

    translated = pipe(
        batches,
        src_lang="en_XX",
        tgt_lang="is_IS",
    )

    example["text"] = " ".join([t["translation_text"] for t in translated])

    # remove chracters repeated more than max_char_repitition times
    example["text"] = re.sub(
        r"(.)\1{%d,}" % (max_char_repitition - 1), r"\1", example["text"]
    )

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

    for split in ds.keys():
        ds[split] = ds[split].map(
            lambda batch: sequential_translate_to_is(
                pipe, batch, batch_size=batch_size
            ),
            batched=False,
        )

    ds.push_to_hub("Sigurdur/imdb-is")

    sentiment = Sentiment()

    ds["train"] = ds["train"].map(
        lambda batch: add_sentiment(batch, sentiment),
        batched=True,
        batch_size=10,
    )

    ds.push_to_hub("Sigurdur/imdb-is-sentiment")
