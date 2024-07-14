import re
import torch
from utils import Sentiment
from datasets import load_dataset
from transformers import pipeline


def translate_to_is(example, batch_size=128, sentiment=None):
    text = example["text"]

    # replace all html tags with a space
    text = re.sub(r"<.*?>", " ", text)
    # remove all non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # remove all whitespace
    text = re.sub(r"\s+", " ", text)
    # text = text.split()
    # batches = [
    #     " ".join(text[i : i + batch_size]) for i in range(0, len(text), batch_size)
    # ]
    translated = pipe(
        text,
        src_lang="en_XX",
        tgt_lang="is_IS",
        batch_size=128,
    )
    cleaned = " ".join([t["translation_text"] for t in translated])

    example["text"] = cleaned

    # add sentiment analysis
    example["emotions"] = sentiment(example["text"])

    print(example)
    print()

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
    sentiment = Sentiment()
    for split in ds.keys():
        ds[split] = ds[split].map(
            lambda batch: translate_to_is(
                batch, batch_size=batch_size, sentiment=sentiment
            ),
            batched=False,
        )

    ds.push_to_hub("Sigurdur/imdb-is")
