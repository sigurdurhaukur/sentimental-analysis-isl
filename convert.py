from datasets import Dataset
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("IMDB-Dataset-GoogleTranslate.csv")
    dataset = Dataset.from_pandas(data)
    print(dataset)
    dataset.push_to_hub("Sigurdur/imdb-isl-google-translate")

    data = pd.read_csv("IMDB-Dataset-MideindTranslate.csv")
    dataset = Dataset.from_pandas(data)
    print(dataset)
    dataset.push_to_hub("Sigurdur/imdb-isl-mideind-translate")
