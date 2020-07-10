
##### Requirements:
All dependancies can be found in `requirements.txt` file.

##### How to run:

1. Install requirements: `pip install -r requirements.txt`

2. ` python main.py --vectors embeddings/arabic-news.bin --train_dataset datasets/tweets/ASTD.csv --score_dataset datasets/tweets/ArTwitter.csv`

##### Command Line Arguments:
- `vectors`:
- `train_dataset`:
- `score_dataset`:

##### Supported Classifiers:
Below is a tentative list of supported classifiers:

- `RandomForestClassifier`
- `SGDClassifier`
- `LinearSVC`
- `NuSVC`
- `LogesticRegression`
- `GaussianNB`

This list will likely be expanded and updated to include additional classifiers that are optimized and have better performance in sentiment classification.


##### Reference:
Embeddings and datasets used are referenced in the paper:

A. Altowayan and L. Tao _"Word Embeddings for Arabic Sentiment Analysis"_, IEEE BigData 2016 Workshop

You can find in the paper in the `Paper` directory.