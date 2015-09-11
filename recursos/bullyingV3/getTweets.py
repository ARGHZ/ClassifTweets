#!/usr/bin/env python

""" 
Sample code of getting tweet JSON objects by tweet ID lists.

You have to install tweepy (This script was tested with Python 2.6 and Tweepy 3.3.0)
https://github.com/tweepy/tweepy
and set its directory to your PYTHONPATH. 

You have to obtain an access tokens from dev.twitter.com with your Twitter account.
For more information, please follow:
https://dev.twitter.com/oauth/overview/application-owner-access-tokens

Once you get the tokens, please fill the tokens in the squotation marks in the
following "Access Information" part. For example, if your consumer key is 
LOVNhsAfB1zfPYnABCDE, you need to put it to Line 33
consumer_key = 'LOVNhsAfB1zfPYnABCDE' 



"""

# call user.lookup api to query a list of user ids.
import warnings
import tweepy
import sys
import json
import codecs
from tweepy.parsers import JSONParser
import csv
import numpy as np

####### Access Information #################

# Parameter you need to specify
consumer_key = 'kxfJjFCXjkRySLkW2aHGeAXxN'
consumer_secret = 'VKalY6au6029H5uqo63VHH1VWcYwaBmlJ36EPulYUBmThyvDUi'
access_key = '1576798795-MJcRA8Yu8nfgDWbIQjshgio6bOoBCBOGZbSOF06'
access_secret = 'jPVa8ELVIDT2StlNJvts6UmZASllsliVdvHg7VikT88ew'

inputFile = 'tweet_id'
outputFile = 'tweet.json'

#############################################
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth_handler=auth, parser=JSONParser())

def contenido_csv(path_archivo):
    # Obtenemos los datos para probar la RNA
    with open(path_archivo, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        data = []
        for row in spamreader:
            data.append(tuple(row))

    return tuple(data)


l, cont = [], 0
metdat_tweets = np.array(contenido_csv('data.csv'))
for row in metdat_tweets[:, :]:
    if row[2] == '"y"' and row[6] != '"victim"':
        cont += 1
        l.append(row[0])
l = np.array(l)
l = np.array_split(l, (float(len(l))/90))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with codecs.open(outputFile, 'w', encoding='utf8') as outFile:
        for row in l:
            sub_l = tuple([int(elem) for elem in row])
            rst = api.statuses_lookup(id_=sub_l)
            for tweet in rst:
                outFile.write(json.dumps(tweet) + '\n')