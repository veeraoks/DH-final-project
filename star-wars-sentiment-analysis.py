#!/usr/bin/env python
# coding: utf-8


#import things 
import nltk 
from nltk.corpus import sentiwordnet as swn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import spacy
import re
import en_core_web_md




nlp = en_core_web_md.load()
lemmatizer = WordNetLemmatizer()




#load data
episodeIV = pd.read_csv('star-wars-movie-scripts/SW_EpisodeIV.txt', delim_whitespace=True, names=["index", "character","dialogue"], header = None)
episodeV = pd.read_csv('star-wars-movie-scripts/SW_EpisodeV.txt', delim_whitespace=True, names=["index", "character", "dialogue"], header = None)
episodeVI = pd.read_csv('star-wars-movie-scripts/SW_EpisodeVI.txt', delim_whitespace=True, names=["index", "character", "dialogue"], header = None)





#create copies so that the original doesn't need to altered
#delete first row as it contained no information

#episode IV
episodeIV_copy = episodeIV.copy()
episodeIV_copy = episodeIV_copy.drop(0)

#episode V
episodeV_copy = episodeV.copy()
episodeV_copy = episodeV_copy.drop(0)

#episode VI
episodeVI_copy = episodeVI.copy()
episodeVI_copy = episodeVI_copy.drop(0)





#let's see which character has the most lines and consider those as the most talkative

top_characters_IV = episodeIV_copy.character.value_counts()
print(top_characters_IV[:4])
top_characters_V = episodeV_copy.character.value_counts()
print(top_characters_V[:4])
top_characters_VI = episodeVI_copy.character.value_counts()
print(top_characters_VI[:4])

#here we will make a judgement call and focus on Luke, Han and C-3PO cause Ben dies in Episode IV and Leia comes along so late





#let's create a new version of the dataframe containing just the three top characters

#episode IV
top_characters_IV_df = episodeIV_copy[(episodeIV_copy['character'] == 'LUKE') | (episodeIV_copy['character'] == 'HAN') | (episodeIV_copy['character'] == 'THREEPIO')]

#episode V
top_characters_V_df = episodeV_copy[(episodeV_copy['character'] == 'LUKE') | (episodeV_copy['character'] == 'HAN') | (episodeV_copy['character'] == 'THREEPIO')]

#episode VI
top_characters_VI_df = episodeVI_copy[(episodeVI_copy['character'] == 'LUKE') | (episodeVI_copy['character'] == 'HAN') | (episodeVI_copy['character'] == 'THREEPIO')]





#let's combine the dataframes
top_characters_all_temp = top_characters_IV_df.append(top_characters_V_df)
top_characters_all = top_characters_all_temp.append(top_characters_VI_df)





#putting the dialogue through a spacy nlp pipeline (for easier manual inspection etc.)
top_characters_all['tokenized'] = top_characters_all['dialogue'].apply(lambda text: nlp(text))





#creating dataframes for each top character 
luke = top_characters_all[(top_characters_all['character'] == 'LUKE')]
han = top_characters_all[(top_characters_all['character'] == 'HAN')]
threepio = top_characters_all[(top_characters_all['character'] == 'THREEPIO')]





#cleaning the text from excessive punctuation and stopwords using regex
luke_dialogue = list(luke['tokenized'])
han_dialogue = list(han['tokenized'])
threepio_dialogue = list(threepio['tokenized'])


pattern = re.compile(r'[^\w\s]')


luke_dialogue_no_punct = pattern.sub('', str(luke_dialogue))
luke_dialogue_clean = ' '.join([word for word in luke_dialogue_no_punct.split() if word not in stopwords.words("english")])



han_dialogue_no_punct = pattern.sub('', str(han_dialogue))
han_dialogue_clean = ' '.join([word for word in han_dialogue_no_punct.split() if word not in stopwords.words("english")])


threepio_dialogue_no_punct = pattern.sub('', str(threepio_dialogue))
threepio_dialogue_clean = ' '.join([word for word in threepio_dialogue_no_punct.split() if word not in stopwords.words("english")])





#tagging
luke_tagged = nltk.pos_tag(word_tokenize(str(luke_dialogue_clean)))
han_tagged = nltk.pos_tag(word_tokenize(str(han_dialogue_clean)))
threepio_tagged = nltk.pos_tag(word_tokenize(str(threepio_dialogue_clean)))





#let's transform the tags into correct ones
def senti_tags(text):
    text_senti = []
    
    for word, tag in text:
        if tag.startswith('NN'):
            tag = 'n'
        elif tag.startswith('VB'):
            tag = 'v'
        elif tag.startswith('RB'):
            tag = 'r'
        elif tag.startswith('JJ'):
            tag = 'a'
        else:
            tag = ''
        if tag != '':
            text_senti.append((word, tag))
            
    return text_senti





#function to retrieve the first sense from wordnet
def synset_senses(text):
    synsets = []
    wn_lemmas = set(wn.all_lemma_names())
    for word, tag in text:
        lemma = lemmatizer.lemmatize(word, tag)
        if lemma in wn_lemmas:
            words = (list(swn.senti_synsets(word))[0])
            synsets.append(words)
    return synsets





#function to calculate the sentiment scores
def sentiment_scores(text):
    new_tags = senti_tags(text)
    synsets = synset_senses(new_tags)
    positives = []
    negatives = []
    neutrals = []
    for syn in synsets:
        score = syn.pos_score() - syn.neg_score()
        if score > 0:
            positives.append(score)
        elif score < 0:
            negatives.append(score)
        else:
            neutrals.append(score)
    #let's get the percentages as the characters have different amount of dialogue
    return (len(positives)/len(synsets))*100, (len(negatives)/len(synsets))*100, (len(neutrals)/len(synsets))*100





#counting the scores and assigning them to variables
luke_sentiment_score = sentiment_scores(luke_tagged)
han_sentiment_score = sentiment_scores(han_tagged)
threepio_sentiment_score = sentiment_scores(threepio_tagged)





#creating dataframes from the results to export and use for visualization
data_df = {'Name': ['Luke','Han', 'Threepio'],'Positive':[luke_sentiment_score[0], han_sentiment_score[0],
                                                          threepio_sentiment_score[0]], 
           'Negative':[luke_sentiment_score[1], han_sentiment_score[1], threepio_sentiment_score[1]], 
                       'Neutral': [luke_sentiment_score[2], han_sentiment_score[2], threepio_sentiment_score[2]]}
scores_df = pd.DataFrame(data=data_df)


#EXAMPLES


#function to give the sense and sentiment score 
# prints out the word and its positive, negative and neutral score respectively 
def sentiments(text):
    new_tags = senti_tags(text)
    synsets = synset_senses(new_tags)
    sentis_scores= []
    for syn in synsets:
        sentis_scores.append((syn, syn.pos_score(), syn.neg_score(), syn.obj_score()))
    return sentis_scores   


#a random example from Luke's dialogue
sentiments(luke_tagged[1924:2024])

#a random example from Han's dialogue
sentiments(han_tagged[1000:1100])

#a random example from Threepio's dialogue
sentiments(threepio_tagged[500:600])

#examples from Han's dialogue
han_line = 'Yeah, great at getting us into trouble.'
han_line_2 = 'Nice work. Great, Chewie! Great! Always thinking with your stomach.'
han_line_3 = "Don't everyone thank me at once."

print(sentiments(nltk.pos_tag(word_tokenize(han_line.lower()))))
print(sentiments(nltk.pos_tag(word_tokenize(han_line_2.lower()))))
print(sentiments(nltk.pos_tag(word_tokenize(han_line_3.lower()))))
