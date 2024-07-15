# Icelandic Sentimental Analysis

Related works: https://skemman.is/bitstream/1946/46224/1/FinalReport.pdf

- their model https://huggingface.co/Birkir/electra-base-igc-is-sentiment-analysis

Lexicon data: https://arxiv.org/pdf/2005.05672, https://arxiv.org/abs/2005.05672
Translation model: https://huggingface.co/mideind/nmt-doc-en-is-2022-10
IMDB dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
Imdb translated: https://github.com/cadia-lvl/sentiment-analysis/
Stop words: https://repository.clarin.is/repository/xmlui/handle/20.500.12537/124?locale-attribute=is
Lemmatization: https://github.com/haukurb/ReynirPackage

## Method

1. Download lexicon data from [here](https://arxiv.org/pdf/2005.05672).
2. Preprocess the data. By deduplicating and removing words that are not in the Icelandic language. Lemmatizing the words using the ReynirPackage and removing stop words using the list from [here](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/124?locale-attribute=is).
3. Download translated IMDB dataset from [here](github.com/cadia-lvl/sentiment-analysis/).

The classes were calculated by lemmatizing the translated IMDB dataset and removing stop words. Then each word was checked against the lexicon data and the average sentiment score was calculated for each review. The classes were then calculated by calculating which class was the closest to it's maximum sentiment score.

TODO:

finetune a transformer model on the translated IMDB dataset. Use active learning to improve the model.
compare to the model from https://huggingface.co/Birkir/electra-base-igc-is-sentiment-analysis
