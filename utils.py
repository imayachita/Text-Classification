#Author: Inneke Mayachita
#this file contains the general functions

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
import seaborn as sns
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk import word_tokenize
from collections import Counter

def clean_specialLetters(cell):
    """
    Cleaning out special characters and non-unicode characters.

    Args:
        cell (str): input string
    Returns:
        clean (str): cleaned string
    """
    removed = re.sub('[^A-Za-z0-9]+', ' ', cell)
    clean = removed.encode("ascii", errors="ignore").decode()
    return clean


def load_stopwords(stopwords_file):
    """
    Load stopwords file

    Args:
        stopwords_file (str): .txt file that contains all stop words
    Returns:
        stopwords (list): stop words in a list
    """
    f = open(stopwords_file,'r',encoding='utf-8')
    stopwords = f.read().split('\n')
    f.close()
    return stopwords


def remove_stopwords(text,stopwords_file):
    """
    Removing stopwords from the text

    Args:
        text (list): input string in list
        stopwords_file (str): .txt file that contains all stop words
    Returns:
        filtered_words (list): cleaned string in list
    """

    stopwords = load_stopwords(stopwords_file)
    filtered_words = []
    for sentence in text:
        tokenized = word_tokenize(sentence)
        cleaned = [word for word in tokenized if word not in stopwords]
        cleaned = ' '.join(word for word in cleaned)
        filtered_words.append(cleaned)

    return filtered_words


def remove_numbers(cell):
    """
    Cleaning out numbers.

    Args:
        cell (str): input string
    Returns:
        clean (str): cleaned string
    """
    removed = re.sub('[0-9]+', '', cell)
    return removed


def clean_data(texts,stopwords_file=None):
    """
    Clean the text data by removing stopwords, numbers, and special characters

    Args:
        texts (list): input string in list
        stopwords_file (str): .txt file that contains all stop words
    Returns:
        clean_data (list): cleaned string in list
    """
    cleaned_data=[]
    for text in texts:
        pro_text = text.casefold()
        pro_text = clean_specialLetters(pro_text)
        pro_text = remove_numbers(pro_text)
        cleaned_data.append(pro_text)

    if stopwords_file is not None:
        cleaned_data = remove_stopwords(cleaned_data,stopwords_file)
    return cleaned_data


def load_data(train_dir,test_dir,sep=''):
    """
    Load data from csv files

    Args:
        train_dir (str): train data file
        test_dir (str): test data file
        sep: separator

    Returns:
        train (dataframe): train data
        test (dataframe): test data

    """
    train = pd.read_csv(train_dir,sep=sep)
    test = pd.read_csv(test_dir)
    return train,test


def plot_barplot():
    """
    plot class distribution
    """
    sns.barplot(x=train['annotations'].value_counts().index,y=train['annotations'].value_counts())


def convert_to_categorical(df_col):
    """
    convert data to categorical
    used to convert data to label

    Args:
        df_col (dataframe column): data to convert

    Returns:
        array of labels in numbers instead of text
    """
    return pd.get_dummies(df_col).to_numpy()


def vectorize(train,test,vectorizer='count',stopwords_file=None):
    """
    vectorize the text

    Args:
        train (dataframe): train data with 2 columns "text" and "annotations"
        test (dataframe): test data with 2 columns "text" and "annotations"
        vectorizer: type of vectorizer, can be either 'count' or 'tfidf'
        stopwords_file: txt file that contains stopwords

    Returns:
        X (array): training data
        Y (array): label training data
        test_X (array): test data
        test_Y (array): label test data
        num_words: the length of vocabulary

    """
    train = pd.read_csv(train)
    test = pd.read_csv(test)

    data = pd.concat([train,test],axis=0)
    print('Number of data: ', len(data))
    corpus = data['text']

    if stopwords_file is not None:
        stopwords=load_stopwords(stopwords_file)
    else:
        stopwords=None

    assert vectorizer == 'tfidf' or vectorizer == 'count', 'vectorizer must be either tfidf or count'
    if vectorizer=='tfidf':
        vectorizer = TfidfVectorizer(strip_accents='ascii',
                                     lowercase=True,
                                     stop_words=stopwords,
                                     min_df=3)
    elif vectorizer=='count':
        vectorizer = CountVectorizer(strip_accents='ascii',
                                 lowercase=True,
                                 stop_words=stopwords,
                                 min_df=3)

    #fit vectorizer to the whole corpus and transform the train and test set
    vectorizer.fit(corpus)
    X = vectorizer.transform(train['text']).toarray()
    num_words = X.shape[1]
    print('num words: ', num_words)
    test_X = vectorizer.transform(test['text']).toarray()

    #convert labels to categorical
    Y = convert_to_categorical(train['annotations'])
    test_Y = convert_to_categorical(test['annotations'])
    return X,Y,test_X,test_Y,num_words


def tokenize(train,test,stopwords_file=None,maxlen=100):
    """
    tokenize the text

    Args:
        train (csv file): train data with 2 columns "text" and "annotations"
        test (csv file): test data with 2 columns "text" and "annotations"
        stopwords_file: txt file that contains stopwords
        maxlen = max sequence length

    Returns:
        X (array): training data
        Y (array): label training data
        test_X (array): test data
        test_Y (array): label test data
        num_words: the length of vocabulary

    """
    train = pd.read_csv(train)
    test = pd.read_csv(test)

    data = pd.concat([train,test],axis=0)
    print('Number of data: ', len(data))
    clean = clean_data(data['text'],stopwords_file)

    #fit tokenizer on the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean)

    num_words = len(tokenizer.word_index)
    print('Number of words in vocab: ', num_words)

    #tokenize data
    X = tokenizer.texts_to_sequences(train['text'])
    X = tokenizer.texts_to_matrix(train['text'])
    X = pad_sequences(X,maxlen=maxlen)

    # test_X = tokenizer.texts_to_sequences(test['text'])
    test_X = tokenizer.texts_to_matrix(test['text'])
    test_X = pad_sequences(test_X,maxlen=maxlen)

    #convert labels to categorical
    Y = convert_to_categorical(train['annotations'])
    test_Y = convert_to_categorical(test['annotations'])
    print('shape X: ', X.shape)
    print('shape Y: ', Y.shape)

    return X,Y,test_X,test_Y,num_words


def tf_igm_vectorizer(train,test,stopwords_file,mode='no_seq'):
    '''
    Implement term-weighting with tf-igm based on paper https://doi.org/10.1016/j.eswa.2016.09.009 by Chen, Kewen., et al
    Args:
        train (csv file): train data with 2 columns "text" and "annotations"
        test (csv file): test_data with 2 columns "text" and "annotations"
        stopwords_file: txt file that contains stopwords
        mode: 'no_seq' represents each input as matrix, not sequence
    Returns:
        X (array): training data
        Y (array): label training data
        test_X (array): test data
        test_Y (array): label test data
        num_words: the length of vocabulary

    '''
    train = pd.read_csv(train)
    clean = clean_data(train['text'],stopwords_file)
    labels = set(train['annotations'].values)
    num_labels = len(labels)
    dataset = []
    t = Tokenizer()
    t.fit_on_texts(clean)

    assert len(clean) == len(train),'Data is truncated'
    encoded = t.texts_to_sequences(clean)

    for i,sentence in enumerate(encoded):
        dataset.append((sentence,train['annotations'].iloc[i]))

    counter_list = []
    for label in labels:
        label_dic = list(filter(lambda x: x[1]==label, dataset))
        label_dic = [e[0] for e in label_dic]
        label_dic = [e for element in label_dic for e in element]
        label_dic = Counter(label_dic)
        counter_list.append(label_dic)

    vocab = len(t.word_index)
    word_freq_dic = {}
    for word_idx in range(1,vocab+1):
        word_freq = []
        for i in range(len(labels)):
            word_freq.append((counter_list[i][word_idx]))
        word_freq_dic[word_idx] = word_freq


    igm_dict = {}
    wg_dict = {}
    alpha = 7
    for word_idx in range(1,vocab+1):
        sorted_list = sorted(word_freq_dic[word_idx],reverse=True)
        denom = 0
        for i in range(num_labels):
             denom += sorted_list[i]*(i+1)
        igm_dict[word_idx] = sorted_list[0]/denom
        wg_dict[word_idx] = 1+(alpha*igm_dict[word_idx])


    idx_to_count = [e[0] for e in dataset]
    idx_to_count = Counter([e for element in idx_to_count for e in element])
    tf_term = {}
    for word_idx in range(1,vocab+1):
        tf_term[word_idx] =  idx_to_count[word_idx] / vocab

    tf_igm = {}
    for word_idx in range(1,vocab+1):
         tf_igm[word_idx] = tf_term[word_idx]*wg_dict[word_idx]


    test = pd.read_csv(test)
    test_clean = clean_data(test['text'],stopwords_file)
    test_encoded = t.texts_to_sequences(test_clean)

    if mode == 'no_seq':
        train_X = t.texts_to_matrix(clean,mode='count')
        test_X = t.texts_to_matrix(test_clean,mode='count')

        for i,sentence in enumerate(train_X):
            for j,word in enumerate(sentence):
                if word==0:
                    train_X[i][j] = 0
                else:
                    train_X[i][j] = tf_igm[j]

        for i,sentence in enumerate(test_X):
            for j,word in enumerate(sentence):
                if word==0:
                    test_X[i][j] = 0
                else:
                    test_X[i][j] = tf_igm[j]
        else:
            train_X = [[tf_igm[x] for x in line] for line in encoded]
            train_X = pad_sequences(train_X,dtype='float32',maxlen=100,padding='post')
            test_X = [[tf_igm[x] for x in line] for line in test_encoded]
            test_X = pad_sequences(test_X,dtype='float32',maxlen=100,padding='post')


    Y = convert_to_categorical(train['annotations'])
    test_Y = convert_to_categorical(test['annotations'])

    return train_X, Y, test_X, test_Y, vocab
