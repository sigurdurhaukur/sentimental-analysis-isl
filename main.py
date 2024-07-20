import re
from utils import Sentiment
from datasets import load_dataset


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

def add_sentiment(batch, sentiment):
    texts = preprocess_text(batch["review"])
    sentiments = sentiment(texts)
    batch["sentiment"] = sentiments

    return batch

if __name__ == "__main__":
    ds = load_dataset("Sigurdur/imdb-isl-google-translate", keep_in_memory=True)
    
    # for testing
    # ds["train"] = ds["train"].select(range(150))
    sentiment = Sentiment()

    ds["train"] = ds["train"].map(
        lambda batch: add_sentiment(batch, sentiment),
        batched=True,
        batch_size=10,
    )

    for i in range(len(ds["train"])):
        if ds["train"]["sentiment"][i] == ["Unknown"]:
            continue

        print(ds["train"]["sentiment"][i])
        print(ds["train"]["review"][i][:200])


    ds.push_to_hub("Sigurdur/imdb-is-sentiment")

