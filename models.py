import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Conv1D, Flatten, MaxPool1D, Embedding, SpatialDropout1D,LSTM, Bidirectional
from keras.callbacks import EarlyStopping, TensorBoard

from gensim.models import FastText
import json
from utils import vectorize,tokenize,tf_igm_vectorizer

class TextCatModel:
    def __init__(self,config_file):
        if config_file is not None:
            with open(config_file) as f:
                config = json.load(f)
                #print(config)

        self.model_type = config['model_type']
        print('model type: ', self.model_type)
        assert self.model_type == "ML" or self.model_type == "FCNN" or self.model_type == "BILSTM" or self.model_type == "CNN"
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.vectorizer = config['vectorizer']
        self.stopwords_file = config['stopwords_file']
        self.maxlen = config['maxlen']
        self.train = config['train_file']
        self.test = config['test_file']

        self.X,self.Y,self.test_X,self.test_Y,self.num_words = self.preprocess(self.train,self.test)

        if self.model_type == "ML":
            self.model = ML_models()
        elif self.model_type == "FCNN":
            self.model = FCNN(self.X,self.Y)
        elif self.model_type == "BILSTM":
            self.model = BILSTM(self.X,self.Y,self.num_words,self.maxlen)
        elif self.model_type == "CNN":
            self.model = CNN(self.X,self.Y,self.num_words,self.maxlen)


    def preprocess(self,train,test):
        if self.model_type == "ML":
            if self.vectorizer == 'tf-igm':
                X,Y,test_X,test_Y,num_words = tf_igm_vectorizer(train,test,stopwords_file=self.stopwords_file)
            else:
                X,Y,test_X,test_Y,num_words = vectorize(train,test,vectorizer=self.vectorizer,stopwords_file=self.stopwords_file)
            Y = np.argmax(Y,axis=1)
            test_Y = np.argmax(test_Y,axis=1)
        else:
            if self.vectorizer == 'tf-igm':
                X,Y,test_X,test_Y,num_words = tf_igm_vectorizer(train,test,stopwords_file=self.stopwords_file)
            else:
                X,Y,test_X,test_Y,num_words = tokenize(train,test,stopwords_file=self.stopwords_file,maxlen=self.maxlen)

        return X,Y,test_X,test_Y,num_words


    def training(self):
        X = self.X
        Y = self.Y
        test_X = self.test_X
        test_Y = self.test_Y
        model = self.model
        if self.model_type == "ML":
             for m in model:
                m.fit(X,Y)
        else:
            tbCallBack = TensorBoard(
                                        log_dir='./Graph',
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images=True
                                        )
            es = EarlyStopping(monitor='val_loss')
            model.fit(
                        X, Y,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose=0,
                        validation_data=(test_X, test_Y),
                        callbacks=[tbCallBack,es]
                       )


    def evaluate(self):
        '''
        evaluate model
        if model is "ML", model is in list with each element is the model
        '''
        test_X = self.test_X
        test_Y = self.test_Y
        model = self.model
        if self.model_type == "ML":
            for m in model:
                pred = m.predict(test_X)
                print('Acc for model', m, ': \n', accuracy_score(test_Y,pred))
                print('Confusion Matrix for model', m, ': \n', confusion_matrix(test_Y, pred))
                print('F1-score for model', m, ': \n', classification_report(test_Y, pred))

        else:
            pred = model.evaluate(test_X, test_Y, verbose=0)
            print('Acc:', pred[1])



#########################################
# LIST OF ALL MODELS

def ML_models():
    #define the models
    LogReg = LogisticRegression(penalty='elasticnet',
                                solver='saga',
                                multi_class='multinomial',
                                class_weight='balanced',
                                l1_ratio=0.5,
                               random_state=1234)

    svc = SVC(kernel='rbf',gamma='auto',decision_function_shape='ovo')
    nb = MultinomialNB()
    rf = RandomForestClassifier(n_estimators=100)
    xgb = XGBClassifier()

    model = [LogReg,svc,nb,rf,xgb]
    return model


def FCNN(X,Y):
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(output_dim=output_dim, activation='sigmoid'))

    model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                 )
    print(model_fcnn.summary())
    return model

def BILSTM(X,Y,num_words,maxlen):
    print(Y.shape)
    output_dim=Y.shape[1]
    inputs = Input(shape=(maxlen,))
    model = Embedding(input_dim=num_words,output_dim=300,trainable=True)(inputs)
    model = Bidirectional(LSTM(100))(model)
    model = Dense(50, activation='relu')(model)
    model = Dense(output_dim=output_dim, activation='sigmoid')(model)
    model_bilstm = Model(inputs=inputs,outputs=model)

    model_bilstm.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                 )
    print(model_bilstm.summary())
    return model_bilstm

def CNN(X,Y,num_words,maxlen):
    output_dim=Y.shape[1]
    inputs = Input(shape=(maxlen,))
    model = Embedding(input_dim=num_words,output_dim=300,trainable=True)(inputs)
    model = Conv1D(64,
              kernel_size = 2,
              kernel_initializer = 'glorot_normal',
              bias_initializer='glorot_normal',
              padding='same')(model)
    model = Flatten()(model)
    model = Dense(50, activation='relu')(model)
    model = Dense(output_dim=output_dim, activation='sigmoid')(model)
    model_cnn = Model(inputs=inputs,outputs=model)

    batch_size = 128
    nb_epoch = 4000

    model_cnn.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                 )
    print(model_cnn.summary())
    return model_cnn
