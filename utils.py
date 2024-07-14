import numpy as np
from reynir import Greynir
from tqdm import tqdm
import os
import pandas as pd


class Sentiment:
    def __init__(
        self,
        data_path="./data/is.tsv",
        cache_path="./data/is_clean.tsv",
        stopwords_path="./rmh_filters/IGC_filters_all.txt",
    ):
        assert os.path.exists(data_path), "Data file not found"

        self.df = pd.read_csv("./data/is.tsv", sep="\t")
        self.stopwords = np.loadtxt(stopwords_path, dtype=str, usecols=0)
        self.cache = cache_path

        # for testing reduce the size of the dataset
        self.columns = self.df.columns[1:]
        self.average_from_each_column = [
            self.df[column].mean() for column in self.columns
        ]
        self.max_values = [self.df[column].max() for column in self.columns]

        try:
            self.greynir = Greynir()
        except Exception as e:
            print(f"Failed to initialize Greynir: {e}")

        if os.path.exists(self.cache):
            self.df = pd.read_csv(self.cache, sep="\t")
        else:
            self.clean_df()

    def clean_df(self):
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates(subset="word")
        self.df = self.df.reset_index(drop=True)

        # drop rows that are not words
        rows_to_drop = [
            row for row in [0, 1, 6, 115, 178, 961, 962] if row < len(self.df)
        ]
        self.df = self.df.drop(self.df.index[rows_to_drop], errors="ignore")
        self.df = self.df.reset_index(drop=True)

        # lemmatize the words
        for i in tqdm(range(len(self.df)), desc="Lemmatizing words"):
            self.df.at[i, "word"] = self.lemmatize(self.df.at[i, "word"])

        # remove stopwords
        self.df = self.df[~self.df["word"].isin(self.stopwords)]
        if self.cache:
            self.df.to_csv(self.cache, sep="\t", header=True, index=False)

    def lemmatize(self, txt: str) -> str:
        s = self.greynir.parse_single(txt)
        if s is not None and hasattr(s, "lemmas"):
            return " ".join(s.lemmas) if s.lemmas else txt
        return txt

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = "".join(e for e in text if e.isalnum() or e.isspace())
        text = text.strip()
        text = self.lemmatize(text)

        txt = text.split()
        text = " ".join([word for word in txt if word not in self.stopwords])

        return text

    def sentence_score(self, sentence: str) -> list:
        sentence = self.preprocess_text(sentence)
        words = sentence.split()

        valid_scores = []
        word_scores = self.df.set_index("word")

        for word in words:
            try:
                if word in word_scores.index:
                    score = (
                        word_scores.loc[word].values[1:].tolist()
                    )  # first column is the word itself
                    valid_scores.append(score)
            except Exception as e:
                print(f"Error processing word '{word}': {e}")
                continue

        if valid_scores:
            scores = np.mean(np.array(valid_scores), axis=0).tolist()
        else:
            # Handle case where no valid scores are found
            scores = [0] * len(self.columns)

        return scores

    def what_class(self, scores: list) -> str:
        # skip valance class, since its a "combination of all classes"
        columns = self.columns[1:]
        scores = scores[1:]

        if scores == [0] * len(scores):
            return "Unknown"

        # take the difference between the score and the max value
        diff = [max_value - score for max_value, score in zip(self.max_values, scores)]
        # take the index of the max difference
        return columns[diff.index(max(diff))]

    def __call__(self, txt):
        if isinstance(txt, str):
            scores = self.sentence_score(txt)
            return self.what_class(scores)
        if isinstance(txt, list):
            return [self.what_class(self.sentence_score(sentence)) for sentence in txt]


if __name__ == "__main__":
    sentiment_data = Sentiment(cache_path="./data/is_clean.tsv")

    sentence = "Ég leigði AM AM CURIOUSYELLOW af myndbandaleigunni minni vegna deilnanna sem urðu í kringum hana þegar hún kom fyrst út árið 1967. Ég heyrði líka að í fyrstu hafi bandarískir siðir lagt hald á hana og hún hafi reynt að komast inn í landið og þar af leiðandi hafi aðdáendur kvikmynda verið umdeildir. Ég varð að sjá þetta sjálf. Sagan fjallar um ungan sænskan leiklistarnema sem heitir Lena og vill læra allt sem hún getur um lífið. Sérstaklega vill hún beina athyglinni að því hvernig venjulegur Svíi hugsar um ákveðin pólitísk málefni eins og Víetnamstríðið og kynþáttahatur í Bandaríkjunum. Á milli þess að spyrja stjórnmálamenn og almenna borgara í Stokkhólmi út í málefni sín. Hún hefur skoðanir á stjórnmálum og hefur kynmök við leiklistarkennara, bekkjarfélaga sína og gifta karlmenn. Það sem fer um mig í myndinni er að fyrir 40 árum var þetta talið klámfengið. Kynlífs- og nektaratriðin eru sjaldgæf og það er langt á milli þeirra. Kynlífs- og nektaratriðin eru þó ekki eins og klámfengið klámfengið myndefni. Samlöndum mínum finnst það hneykslanlegt að sjá að kynlíf og nekt skuli vera áberandi í sænskum kvikmyndum. Jafnvel Ingmar Bergman er sannfærður um að svar þeirra við því hvernig gamli góði John Ford brást við kynlífssenum í myndum sínum. Ég hrósa kvikmyndagerðarfólki fyrir að allt kynlíf sem sýnt er í myndinni er sýnt í listrænum tilgangi frekar en til að hneyksla fólk og græða á því að vera sýnt í klámmyndahúsum í Ameríku. ÉG GET GERT KLÓVAK Þetta er góð mynd fyrir alla sem vilja kynna sér kjötið og kartöflurnar án þess að vera með einhvern refil sem er ætlaður sænskum kvikmyndahúsum. En í raun og veru er ekki mikið pælt í þessari mynd."

    scores = sentiment_data.sentence_score(sentence)
    print(scores)
    print(sentiment_data.what_class(scores))
