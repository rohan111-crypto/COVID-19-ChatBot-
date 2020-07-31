#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this Ai chatbot


# In[1]:


#Import libraries
from newspaper import Article
import random
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings('ignore')
nltk.download('punkt',quiet=True)


# In[2]:


#Get the article
article = Article('https://en.wikipedia.org/wiki/COVID-19_pandemic')
article.download()
article.parse()
article.nlp()
corpus=article.text


# In[3]:


#Print the article text
print(corpus)


# In[4]:


#Tokenization
text=corpus
sentence_list=nltk.sent_tokenize(text) #gives us a list of sentences


# In[5]:


#Print the list of sentences
print(sentence_list)


# In[6]:


# A function to return a greeting response to a user's greeting
def greetingresponse(text):
    text=text.lower()
    #bot greeting
    bot_greetings = ['hello!','hi!']
    #user greetings
    user_greetings = ['hi','hey','hello','wassup','namaste','hola']
    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)


# In[7]:


def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0,length))
    x=list_var
    for i in range(length):
        for j in range(length):
            if(x[list_index[i]]>x[list_index[j]]):
                temp=list_index[i];
                list_index[i]=list_index[j];
                list_index[j]=temp
                
    return list_index


# In[8]:


#create the bots response
def bot_response(user_input):
    user_input=user_input.lower()
    sentence_list.append(user_input)
    bot_response=''
    cm = CountVectorizer().fit_transform(sentence_list)
    similarity_scores = cosine_similarity(cm[-1],cm)
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)
    index = index[1:]
    response_flag = 0
    j=0
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            bot_response=bot_response+' '+sentence_list[index[i]]
            response_flag=1
            j=j+1
            if j>2:
                break
                
    if response_flag==0:
        bot_response=bot_response+' '+"i apologize, i don't understand"
        
    sentence_list.remove(user_input)
    
    return bot_response


# In[10]:


#start the chat
print('Covid Guide : Hello! I am your guide, ask me any queries about COVID 19. If u want to exit type bye.')
exit_list=['bye','tata','sayonara','exit','thats all','quit','see you later','see u later','break']
while(True):
    user_input=input()
    if user_input.lower() in exit_list:
        print('Guide : Happy to help anytime!')
        break
    else:
        if greetingresponse(user_input)!=None:
            print('Guide : '+greetingresponse(user_input))
        else:
            print('Guide : '+bot_response(user_input))
    


# In[ ]:




