#!/bin/python2
from __future__ import print_function
from gensim.parsing.preprocessing import strip_non_alphanum, preprocess_string
from gensim.corpora.dictionary import Dictionary
from keras.models import load_model
import numpy as np
import os
import subprocess
import time

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


try:
    input = raw_input
except NameError:
    pass

try:
    model = load_model('model/model_nn.h5')
except IOError:
    if 'model_nn.tar.gz' not in os.listdir('model'):
        raise IOError("Could not find Sentiment Analysis model. Ensure model is present in: ./model")
    else:
        process = subprocess.Popen("cd model/; "\
                                   "tar -zxf model_nn.tar.gz; cd ..",
                                   shell=True, stdout=subprocess.PIPE)
        process.wait()
        model = load_model('model/model_nn.h5')
vocab = Dictionary.load('model/sent_model_vocab_model')

def predict(text):
    preprocessed = [word[:-3] if word[-3:] == 'xxx' else word for word in
                    preprocess_string(text.replace('not', 'notxxx'))]
    text_col = [(vocab.token2id[word] + 1) for word in preprocessed
                if word in vocab.token2id.keys()]
    text_col = [text_col]
    tweet_l = 20
    if len(text_col[0]) < tweet_l:
        for i in range(tweet_l - len(text_col[0])):
            text_col[0].append(0)
    elif len(text_col[0]) > tweet_l:
        while len(text_col[-1]) > tweet_l:
            text_col.append(text_col[-1][tweet_l:])
            text_col[-2] = text_col[-2][:tweet_l]
    prediction = 0
    for txt in text_col:
        prediction += model.predict(np.array([txt]), batch_size=1)
    prediction /= len(text_col)
    return prediction

ending = 'It was extremely decent conversing with you and I trust that now you'\
           'feel better subsequent to conversing with me.\nBest of fortunes for your future'\
           'Bye!'

def frnd():
    reply = input('How are your companions getting together with your desires?'\
                     '\n')
    if(predict(reply) >=0.4):
        reply = input('Did you break up recently?\n')
        if(predict(reply)>=0.4):
            print(name + ", try not to feel tragic. Take as much time as is needed and mend properly,"\
                        "take a gander at what's happened, gain from it, and discover approaches to "\
                        "assemble another and sound life.\nAll any of us needs is to "\
                        "be upbeat. For a few, this requires the ideal individual to "\
                        "be our other half, and for others, it implies finishing "\
                        "the condition yourself. In any case, to locate the right "\
                        "individual, you should be the ideal individual. What's more, assume that "\
                        "over the long haul, your endeavors will prompt your own "\
                        "individual cheerful completion.")
            print(ending)
        else:
            print(name + ", try not to stress. You might be at a point where comparable "\
                        "individuals are not in your life at this moment. That occurs in "\
                        "life from time to time.\nIt is smarter to be far from "\
                        "incongruent individuals and those individuals are pulled in to "\
                        "you when you put on a show to be somebody you aren't.\nBe as "\
                        "diverse as you really seem to be, become more acquainted with yourself at a "\
                        "profound level, regard your uniqueness, connect with "\
                        "individuals truly, and in the long run the general population who acknowledge "\
                        "you will see and be attracted.")
            print(ending)
    else:
        print("Many individuals have a tendency to expect excessively of others, their family, "\
              "their companions or even just colleagues. It's a typical mistake"\
              ", individuals don't think precisely the way you do.\nDon't let the "\
              "suppositions of others influence you to overlook what you merit. You are "\
              "not in this world to satisfy the desires of others, "\
              "nor should you feel that others are here to satisfy yours."\
              "\nThe initial step you should take in the event that you need to figure out how to "\
              "quit expecting excessively from individuals is to just acknowledge and "\
              "acknowledge the way that no one is flawless and that everybody "\
              "commits errors once in a while.")
        print(ending)
        
def family():
    print(name + ", try not to take excessively pressure. You should simply change "\
                "your needs. Try not to go up against pointless obligations and "\
                "responsibilities.\nTake counsel from individuals whose feeling you "\
                "trust, and get particular counsel when issues arise.\nYou should "\
                "utilize pressure administration strategies and dependably seek after the best. "\
                "These circumstances emerge in everybody's life and what is important the "\
                "most is taking the correct choice at such minutes.")
    print(ending)

def work():
    print(name + ", try not to take excessively pressure. I can show some extremely cool "\
                  "approaches to deal with it.\nYou ought to create sound reactions which "\
                  "incorporate doing standard exercise and taking great quality rest. "\
                  "You ought to have clear limits between your work or scholastic "\
                  "life and home life so you ensure that you don't blend them.\n"\
                  "Techniques, for example, contemplation and profound breathing activities can be "\
                  "truly helping in mitigating stress.\n Always set aside opportunity to "\
                  "revive to dodge the negative impacts of endless pressure "\
                  "what's more, burnout. We require time to recharge and come back to our pre-"\
                  "feeling of anxiety of working.")
    print(ending)

def sorrow1():
    reply = input('I get it. Appears as though something\'s annoying you.'\
                     'Might you be able to additionally portray it, in short?\n')
    if(predict(reply)>=0.4):
        reply = input('It appears like however, the issue may be a bit '\
                         'troubling, it may not really be intense. '\
                         'What are your musings on this?\n')
        if(predict(reply)>=0.5):
            reply = input('It would appear that you concur with me. Wanna sign off?\n')
            if(predict(reply)>0.55):
                print("That is alright. It was pleasant conversing with you. You can talk "\
                      "with me whenever you want.\nBye " + name + "!")
            else:
                sorrow3() 
        else:
            sorrow3()
    else:
        sorrow2()

def sorrow2():
    reply = input('It would be ideal if you dont hesitate to share your emotions ' + name +\
                     ', consider me your friend.\n')

    if(predict(reply)>=0.3):
        reply = input('I see. Among the musings happening in your psyche, '\
                         'which one miracles you the most?\n')
        reply = input('For what reason do you think it upsets you?\n')
        print("Approve. You simply distinguished what we call a programmed thought. "\
              "Everybody has them. They are contemplations that instantly fly to "\
              "mind with no exertion on your part.\nMost of the time the "\
              "thought happens so rapidly you don't see it, however it has a "\
              "affect on your feelings. It's normally the feeling that you "\
              "see, instead of the thought.\nOften these programmed "\
              "considerations are contorted somehow yet we for the most part don't stop "\
              "to scrutinize the legitimacy of the idea. In any case, today, that is "\
              "what we will do.")
        reply = input('So, ' + name + ', are there signs that opposite '\
                         'could be true?\n')

        if(predict(reply)>=0.4):
            print("I'm happy that you understood that the inverse could be "\
                  "genuine. The reason these are called 'false convictions' is "\
                  "since they are extraordinary methods for seeing the world. "\
                  "They are dark or white and disregard the shades of dim in "\
                  "between.\nNow that you have found out about this cool "\
                  "strategy, you can apply it on a large portion of the issues that "\
                  "you will confront. On the off chance that despite everything you feel stuck anytime, you "\
                  "can simply visit with me.\nBest of fortunes for your future "\
                  "attempts. Bye!")
        else:
            sorrow4()
    else:
        sorrow4()

def sorrow3():
    reply = input('Feel agreeable. Would you be able to quickly clarify about your '\
                     'day?\n')
    reply = input('What are the exercises that make up your a large portion of the '\
                     'day?\n')
    reply = input('It would appear that you may feel great talking '\
                     'about yourself. Might you be able to share your feelings?\n')
    if(predict(reply)>=0.3):
        sorrow2()
    else:
        sorrow4()
    
def sorrow4():
    print("My sensitivities. It would appear that it may be a state of concern. Don't "\
          "stress, that is what I'm here for!")
    reply_frnd = input('How are things going ahead with your friends?\n')
    reply_family  = input('How is your association with your parents?\n')
    reply_worklife = input('How is your professional or scholastic life going on?\n')
    if(predict(reply_frnd)<=0.3):
        frnd()
    else:
        if(predict(reply_family)<=0.3):
            family()
        else:
            work()

print('\n\nHi! A debt of gratitude is in order for coming here. I am a chatbot. Individuals say that''I am a kind and receptive bot.')
name = input('If its not too much trouble let me know your name.\n')
try:
    preprocessed = [word for word in preprocess_string(name) if word not in (
                    'people', 'call', 'friend')][0]
    name = [word for word in strip_non_alphanum(name.lower()).split(
            ) if preprocessed in word][0]
except:
    name = name.split()[0]
name = name[0].upper() + name[1:]
print("Hi " + name + "! My name's SYVBot. How about we begin with our session.")
reply = input("Hows life?\n")
if (predict(reply) >= 0.55):
    reply = input('That is great. Are you as a rule this glad, or are there '\
                     'a few stresses that you need to talk about?\n') 
    if (predict(reply)>=0.7):
        reply = input('You appear to be extremely content. Wanna sign off?\n')
        if(predict(reply)>=0.7):
            print('Ok, bye ' + name + '!')
        else:
            reply = input('Is there something annoying you? Would you '\
                             'share it with me?\n') 
            if(predict(reply)>=0.7):
                print("That is alright. It was pleasant conversing with you. You can talk "\
                      "with me whenever you want.\n Bye" + name + "!")
            else:
                sorrow1()
    else:
        sorrow1()
else:
    sorrow3()



