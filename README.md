# Text-Classification

Text Classification with Machine Learning and Deep Learning:
ML:
1. Logistic Regression
2. SVM
3. Naive Bayes Classifier
4. Random Forest Classifier
5. Xtreme Gradient Boost Classifier

DL:
1. FCNN
2. CNN
3. BILSTM

Feature Extraction:
1. Count Vectorizer
2. TF-IDF Vectorizer
3. TF-IGM Vectorizer based on this paper https://doi.org/10.1016/j.eswa.2016.09.009 by Chen, Kewen, et.al.,

Use config.json to set the configurations:
```
{
    "maxlen" : [maximum sentence length],
    "model_type" : ["ML" or "FCNN" or "BILSTM" or "CNN"],
    "batch_size" : [batch size],
    "epochs" : [number of epochs],
    "vectorizer" : ["count" or "tfidf" or "tf-igm"],
    "stopwords_file" : [txt file that contains stopwords],
    "train_file" : [csv file that contains training data with 2 columns "text" and "annotations"],
    "test_file" : [csv file that contains training data with 2 columns "text" and "annotations"]
}
```

How to run the code:
```
python main.py
```
