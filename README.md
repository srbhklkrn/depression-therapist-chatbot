Depression Therapist: Chatbot Approach


The dataset being used is obtained from Kaggle. The dataset being obtained is a set of
tweets, classified into positive (4) and negative (0), as the training data, of which 20% is
used as cross validation data. 494 tweets from twitter, classified into positive(4), neutral(2)
and negative(0), as the test data.

The data is preprocessed by using Gensim library which removes hashtags, website links
and user references in the tweets. Vocabulary of words is initialized with Gensim
Dictionary and words replaced with respective position in the vocabulary plus one.
Tweets of length less than 20 were zero-padded to length 20. Those of length greater than
20 were split into tweets of length 20 and the last split part is zero-padded, if necessary.
Zero padding is done for supplying variable length sequences to the LSTM layer.
The twitter dataset is preprocessed, containing 1600000 tweets classified as positive and
negative, then initialized a vocabulary using Gensim Dictionary and replaced words with
positions in the vocabulary. Each tweet is limited to 20 words and split them if they exceed
the limit.

A model is coded in python using Tenserflow, Keras and Gensim libraries. Then
implemented a LSTM in the neural network model for identifying context in the tweet.
This neural network model uses binary cross entropy loss function and the Adam optimizer
with default parameters, but with Nesterov momentum.

A tree structure is created for the responses for the related questions according to the result
obtained after applying sentiment analysis to the user’s answer. This tree structure is used
by the chatbot to generate required responses for the specified queries or conversations.
Merged and integrated the concepts of Cognitive Behavioral Therapy, as learned from
various online resources and books, to the tree structure. Obtained specific psychological
advice from online and offline sources for specific cases of depression. Integrated these
cases as a separate branch of the tree for tackling slightly different cases of depression.
Created the chatbot python file and implemented the tree structure in python. Integrated
sentiment analysis in the chatbot functioning, with thresholds for classification into leaf
nodes tweaked for high accuracy.

Fine-tuned the parameters for the sentiment analysis model for optimum balance between
performance and accuracy.

#### To train model just run

```Python
python sent_model_vocab.py
```
You can also download both pretrained models (`model_nn.h` and `sent_model_vocab_model`) from [here](https://drive.google.com/file/d/1wa9CuGO3y4I-SGDDUfw-qbj2zoGBmfwn/view?usp=sharing)
