#!/bin/python2
import os
import codecs
import sys
import numpy as np
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation,\
                                         stem_text
from gensim.corpora.dictionary import Dictionary
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau,\
                            ModelCheckpoint
import time

def export(type_data='train'):
    print "Extracting data..."
    if type_data.lower() == 'train':
        filename = 'train_data.csv'
    elif type_data.lower() == 'test':
        filename = 'train_data2.csv'
    data_file = codecs.open('tweet_data/' + filename, encoding='ISO-8859-1')
    data = []
    for tweet in data_file.read().split('\n')[:-1]:
        data.append([string for string in tweet.split('"') if string not in [
                    '', ',']])
    data_file.close()
    labels = [(float(tweet[0]) / 4.0) for tweet in data]
    tweets = [tweet[-1] for tweet in data]

    print "Preprocessing data..."
    for i, tweet in enumerate(tweets):
        new_tweet = ' '.join([word for word in tweet.split(' ') if len(word)\
                            > 0 and word[0] not in ['@', '#'] and 'http' not\
                            in word]).strip()
        pro_tweet = [word[:-3] if word[-3:] == 'xxx' else word for word in
                    preprocess_string(new_tweet.replace('not', 'notxxx'))]
        #pro_tweet = preprocess_string(new_tweet)
        if len(pro_tweet) < 2:
            tweets[i] = strip_punctuation(stem_text(new_tweet.lower())).\
                        strip().split()
        else:
            tweets[i] = pro_tweet
        sys.stdout.write("\r%d tweet(s) pre-processed out of %d\r" % (
                        i + 1, len(tweets)))
        sys.stdout.flush()

    print "\nCleaning data..."
    backup_tweets = np.array(tweets)
    backup_labels = np.array(labels)
    tweets = []
    labels = []
    for i, tweet in enumerate(backup_tweets):
        if len(tweet) >= 2:
            tweets.append(tweet)
            labels.append(backup_labels[i])
    del backup_tweets
    del backup_labels

    # Shuffle the dataset
    data = zip(tweets, labels)
    np.random.shuffle(data)
    tweets, labels = zip(*data)

    return (tweets, labels)

def create_vocab(tweets):
    print "Building vocabulary..."
    vocab = Dictionary()    
    vocab.add_documents(tweets)
    vocab.save('vocab_sentiment')
    return vocab

def get_vocab(tweets=None):
    if 'vocab_sentiment' in os.listdir('.'):
        if not tweets:
            print "Loading vocabulary..."
            vocab = Dictionary.load('vocab_sentiment')
            print "Loaded vocabulary"
            return vocab
        response = raw_input('Vocabulary found. Do you want to load it? (Y/n)'\
                             ': ')
        if response.lower() in ['n', 'no', 'nah', 'nono', 'nahi', 'nein']:
            if not tweets:
                tweets, labels = export()
                del labels
            return create_vocab(tweets)
        else:
            print "Loading vocabulary..."
            vocab = Dictionary.load('vocab_sentiment')
            print "Loaded vocabulary"
            return vocab
    else:
        if not tweets:
            tweets, labels = export()
            del labels
        return create_vocab(tweets)

def init_with_vocab(tweets=None, labels=None, vocab=None, type_data='train'):
    if not tweets and not labels:
        tweets, labels = export(type_data)
    elif tweets and labels:
        pass
    else:
        print "One of tweets or labels given, but not the other"
        return
    if not vocab and type_data == 'train':
        vocab = get_vocab(tweets)
    elif not vocab:
        vocab = get_vocab()

    print "Replacing words with vocabulary numbers..."
    #if type_data == 'train':
        #max_tweet_len = max([len(tweet) for tweet in tweets])
    #else:
        #max_tweet_len = 40 #Empirically obtained :P
    max_tweet_len = 20
    numbered_tweets = []
    numbered_labels = []
    for tweet_num, (tweet, label) in enumerate(zip(tweets, labels)):
        current_tweet = []

        for word in tweet:
            if word in vocab.token2id:
                current_tweet.append(vocab.token2id[word] + 1)

        if len(current_tweet) <= max_tweet_len:
            current_tweet_len = len(current_tweet)
            for i in range(max_tweet_len - current_tweet_len):
                current_tweet.append(0)
            numbered_tweets.append(current_tweet)
            numbered_labels.append(label)

        else:
            while len(current_tweet) > max_tweet_len:
                numbered_tweets.append(current_tweet[:max_tweet_len])
                numbered_labels.append(label)
                current_tweet = current_tweet[max_tweet_len:]
            if len(current_tweet) > 1:
                current_tweet_len = len(current_tweet)
                for i in range(max_tweet_len - current_tweet_len):
                    current_tweet.append(0)
                numbered_tweets.append(current_tweet)
                numbered_labels.append(label)

    print "Replaced words with vocabulary numbers"
    del tweets
    labels = np.array(numbered_labels)
    del numbered_labels
    return (numbered_tweets, labels, len(vocab))

def create_nn(vocab_len=None, max_tweet_len=None):
    if vocab_len == None:
        print "Error: Vocabulary not initialized"
        return
    if max_tweet_len == None:
        print "Error: Please specify max tweet length"
        return

    nn_model = Sequential()
    nn_model.add(Embedding(input_dim=(vocab_len + 1), output_dim=32,
                           mask_zero=True))
    nn_model.add(LSTM(128))
    nn_model.add(Dense(32, activation='sigmoid', kernel_regularizer=l2(0.05)))
    nn_model.add(Dropout(0.3))
    nn_model.add(Dense(1, activation='sigmoid'))

    nn_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=[
                     'accuracy'])

    print "Created neural network model"
    return nn_model

def get_nn(vocab_len=None, max_tweet_len=None):
    if 'model_nn.h5' in os.listdir('.'):
        response = raw_input('Neural network model found. Do you want to load'\
                            ' it? (Y/n): ')
        if response.lower() in ['n', 'no', 'nah', 'nono', 'nahi', 'nein']:
            return create_nn(vocab_len, max_tweet_len)
        else:
            print "Loading model..."
            nn_model = load_model('model_nn.h5')
            print "Loaded model"
            return nn_model
    else:
        return create_nn(vocab_len, max_tweet_len)

def train_nn(tweets=None, labels=None, nn_model=None):
    if tweets is None and labels is None:
        tweets, labels, vocab_len = init_with_vocab()
    elif tweets is not None and labels is not None:
        pass
    else:
        print "One of tweets or labels given, but not the other"
        return
    if not nn_model:
        max_tweet_len = max([len(tweet) for tweet in tweets])
        nn_model = get_nn(vocab_len, max_tweet_len)

    # Callbacks (extra features)
    tb_callback = TensorBoard(log_dir='./Tensorboard/' + str(time.time()))
    early_stop = EarlyStopping(monitor='loss', min_delta=0.025, patience=6)
    lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, min_lr=0.00001,
                                patience=2, epsilon=0.1)
    saver = ModelCheckpoint('model_nn.h5', monitor='val_acc')

    try:
        nn_model.fit(tweets, labels, epochs=50, batch_size=8192, callbacks=
                    [tb_callback, early_stop, lr_reducer, saver], 
                    validation_split=0.2)
    except KeyboardInterrupt:
        pass
    nn_model.save('model_nn.h5')
    print "Saved model"
    del tweets
    del labels
    tweets_test, labels_test, _ = init_with_vocab(type_data='test')
    print nn_model.evaluate(tweets_test, labels_test, batch_size=32)

if __name__ == '__main__':
    train_nn()

