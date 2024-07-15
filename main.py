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

    ds.push_to_hub("Sigurdur/imdb-is")

    sentiment = Sentiment()

    ds["train"] = ds["train"].map(
        lambda batch: add_sentiment(batch, sentiment),
        batched=True,
        batch_size=10,
    )

    ds.push_to_hub("Sigurdur/imdb-is-sentiment")
