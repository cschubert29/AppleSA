
# coding: utf-8

# In[1]:


'''
Florida International University
CAP 5771 - Final Project
Spring 2018
Constanza Schubert, Claudio Romano
EDA
'''
#EDA and preprocessing work on both Historical and 'current' data
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[2]:


#read json files
df = pd.read_json('../data/2013-2014.json', encoding='utf-8')
df_current = pd.read_json('../data/2015-2018.json', encoding='utf-8')


# In[3]:


df.head()
df_current.head()


# In[4]:


#Add new column of number of characters in tweet
df['char_length'] = df['text'].str.len()
df_current['char_length'] = df_current['text'].str.len()

#Boxplots showing distribution of number of characters per tweet
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.char_length)
plt.title('Number of characters per tweet: Historical data 2013-2014')
plt.show()
plt.boxplot(df_current.char_length)
plt.title('Number of characters per tweet: Presidential data 2015-2018')
plt.show()


# In[5]:


def clean_html(text):
    soup = BeautifulSoup(text, 'html5lib')    
    souped = soup.get_text()
    return souped
def extract_hashtags(s):
    hashed = set(part[1:] for part in s.split() if part.startswith('#')) 
    return len(hashed)   
#Feature creation: number of urls per tweet
def get_url(u):
    p = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    urls = p.findall(u)
    return len (urls)
#Determine if tweet is actually a quote from another person ("quoted" tweet)
def extract_quotes(q):
    if q.startswith('\"') and q.endswith('\"'):
        is_quote = 'True'
    else:
        is_quote = 'False'
    return is_quote   
#Sentiment Analysis using TextBlob
def clean_tweet(tweet):
    #Utility function to clean the text in a tweet by removing links and special characters using regex
    return ' '.join(re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
def tweet_blob(tweet):
    tweetblob = TextBlob(clean_tweet(tweet))
    return tweetblob
def analize_sentiment(tweet):
    #Utility function to classify the polarity of a tweet using textblob.
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
#Feature creation - Grammatical structure
def upper_case(tweet):
    upper_count = []
    words = tweet_blob(tweet).words
    for word in words:
        if word.isupper():
            upper_count += word
    return len(upper_count)
#Number of proper nouns in tweet
def get_proper_noun(tweet):
    proper_noun = []
    pos_tags = tweet_blob(tweet).tags
    for words, tag in pos_tags:
        if tag == 'NNP' or tag == 'NNPS':
            proper_noun += tag
    return len(proper_noun)
#Number of personal and posessive pronouns in tweet
def get_pronouns(tweet):
    pronouns = []
    pos_tags = tweet_blob(tweet).tags
    for words, tag in pos_tags:
        if tag == 'PRP' or tag == 'PRP$':
            pronouns += tag
    return len(pronouns)
#Number of regular adjectives per tweet
def get_adjectives(tweet):
    adjectives = []
    pos_tags = tweet_blob(tweet).tags
    for words, tag in pos_tags:
        if tag == 'JJ':
            adjectives += tag
    return len(adjectives)
#Number of regular adverbs per tweet
def get_adverbs(tweet):
    adverbs = []
    pos_tags = tweet_blob(tweet).tags
    for words, tag in pos_tags:
        if tag == 'RB':
            adverbs += tag
    return len(adverbs)
#Number of comparative/superlative adjectives per tweet
def get_super_adjectives(tweet):
    super_adjectives = []
    pos_tags = tweet_blob(tweet).tags
    for words, tag in pos_tags:
        if tag == 'JJR' or tag == 'JJS':
            super_adjectives += tag
    return len(super_adjectives)
#Number of comparative/superlative adverbs per tweet
def get_super_adverbs(tweet):
    super_adverbs = []
    pos_tags = tweet_blob(tweet).tags
    for words, tag in pos_tags:
        if tag == 'RBR' or tag == 'RBS':
            super_adverbs += tag
    return len(super_adverbs)
#Longest sentence in a tweet
def max_sentence_length(tweet):
    sent_lengths = []
    sentences = TextBlob(tweet).sentences
    for sentence in sentences:
        words = sentence.words
        for word in words:
            sent_lengths.append(len(words))
    return np.max(sent_lengths)
#Shortest sentence in a tweet
def min_sentence_length(tweet):
    sent_lengths = []
    sentences = TextBlob(tweet).sentences
    for sentence in sentences:
        words = sentence.words
        for word in words:
            sent_lengths.append(len(words))
    return np.min(sent_lengths)

def is_quoted(text):
    if (text.startswith('"') and text.endswith('"')):
        return 1
    else:
        return 0
def has_media(text):
    if 'http' in text:
        return 1
    else:
        return 0


# In[6]:


def format_and_add_features(df_original):
    df_original['SA'] = np.array([ analize_sentiment(tweet) for tweet in df_original['text'] ])
    df_original['text'] = df_original['text'].apply(clean_html)
    df_original['no_hashtags'] = df_original['text'].apply(extract_hashtags)
    df_original['no_url'] = df_original['text'].apply(get_url)
    df_original['is_retweet2'] = df_original['text'].apply(extract_quotes)
    df_original['clean_tweet'] = df_original['text'].apply(clean_tweet)
    df_original['no_words'] = np.array([ len(tweet_blob(tweet).words) for tweet in df_original['text'] ])
    df_original['no_sentences'] = np.array([ len(tweet_blob(tweet).sentences) for tweet in df_original['text'] ])
    df_original['no_uppercase'] = df_original['text'].apply(upper_case)
    df_original['no_proper_noun'] = df_original['clean_tweet'].apply(get_proper_noun)
    df_original['no_pronouns'] = df_original['clean_tweet'].apply(get_pronouns)
    df_original['no_adjectives'] = df_original['clean_tweet'].apply(get_adjectives)
    df_original['no_adverbs'] = df_original['clean_tweet'].apply(get_adverbs)
    df_original['no_super_adjectives'] = df_original['clean_tweet'].apply(get_super_adjectives)
    df_original['no_super_adverbs'] = df_original['clean_tweet'].apply(get_super_adverbs)
    df_original['max_sent_length'] = df_original['text'].apply(max_sentence_length)
    df_original['min_sent_length'] = df_original['text'].apply(min_sentence_length)
    return df_original


# In[7]:


def create_date_features(df_original):
    df_original["AM/PM"] = df_original["created_at"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%p'))
    df_original["Time"] = df_original["created_at"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%H:%M:%S'))
    df_original["Day_of_the_Week"] = df_original["created_at"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%A'))
    df_original["Day_of_the_Month"] = df_original["created_at"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%d'))
    df_original["Month"] = df_original["created_at"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%B'))
    df_original["Year"] = df_original["created_at"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').year)
    df_original["Hour"] = df_original["created_at"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%H'))
    return df_original
def create_grammar_features(df_original):
    df_original['no_exclaim'] = df_original['text'].apply(lambda x: str.count(str(x.encode('utf-8')), '!'))
    df_original['no_question'] = df_original['text'].apply(lambda x: str.count(str(x.encode('utf-8')), '?'))
    df_original['no_semicolons'] = df_original['text'].apply(lambda x: str.count(str(x.encode('utf-8')), ';'))
    df_original['no_colons'] = df_original['text'].apply(lambda x: str.count(str(x.encode('utf-8')), ':'))
    df_original['no_commas'] = df_original['text'].apply(lambda x: str.count(str(x.encode('utf-8')), ','))
    df_original['no_ellipses'] = df_original['text'].apply(lambda x: str.count(str(x.encode('utf-8')), '...'))
    df_original['Quoted'] = df_original['text'].apply(lambda x: is_quoted(x))
    df_original["Media"] = df_original["text"].apply(lambda x: has_media(x))
    return df_original


# In[ ]:


# Formatting both dataframes and adding new features based on original ones
df = format_and_add_features(df)
df_current = format_and_add_features(df_current)


# In[ ]:


df = create_date_features(df)
df_current = create_date_features(df_current)

df = create_grammar_features(df)
df_current = create_grammar_features(df_current)


# In[ ]:


df.head()


# In[ ]:


df_current.columns


# In[ ]:


#lists with classified tweets
pos_tweets = len(df[df['SA'] > 0])/float(len(df))
neu_tweets = len(df[df['SA'] == 0])/float(len(df)) 
neg_tweets = len(df[df['SA'] < 0])/float(len(df))

pos_tweets_current = len(df_current[df_current['SA'] > 0])/float(len(df_current))
neu_tweets_current = len(df_current[df_current['SA'] == 0])/float(len(df_current))
neg_tweets_current = len(df_current[df_current['SA'] < 0])/float(len(df_current))

#print percentages of classified tweets
print("Sentiment Analysis results for historical tweet data 2013-2014")
print("\tPositive tweets: %.2f%%"%(pos_tweets * 100))
print("\tNeutral tweets: %.2f%%"%(neu_tweets * 100))
print("\tNegative tweets: %.2f%%"%(neg_tweets * 100))

print("Sentiment Analysis results for presidential tweet data 2015-2018")
print("\tPositive tweets: %.2f%%"%(pos_tweets_current * 100))
print("\tNeutral tweets: %.2f%%"%(neu_tweets_current * 100))
print("\tNegative tweets: %.2f%%"%(neg_tweets_current* 100))


# In[ ]:


#Hypothesis test for difference in proportions (two-tail test)
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm

#number of successes (positive tweets)
count = [len(df[df['SA'] > 0]), len(df_current[df_current['SA'] > 0])]
#number of trials (total number of tweeets)
nobs = [len(df['text']), len(df_current['text'])]
stat, pval = proportions_ztest(count, nobs, value=None, alternative='two-sided', prop_var=False)
critical_z = norm.ppf(1-(0.05/2))#critical value at 95% confidence
print('The test statistic equals ' + str(stat) + ' and the critical value is ' + str(critical_z))
print('Conclude that there is a statistically significant difference between the proportion of positive tweets from prior presidency announcement and post.')


# In[ ]:


#Count Vectorizer - term frequencies
cvec = CountVectorizer(tokenizer=nltk.word_tokenize,stop_words='english',max_features=10000, min_df=2)

# Train the vectorizer on clean tweets
cvec.fit(df.clean_tweet)

len(cvec.get_feature_names()) # number of words extracted from the tweet corpus

neg_doc_matrix = cvec.transform(df[df.SA == -1].clean_tweet)
pos_doc_matrix = cvec.transform(df[df.SA == 1].clean_tweet)
neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()


term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
print(term_freq_df.sort_values(by='total', ascending=False).iloc[:50])


# In[ ]:


#Visualizations from CountVectorizer

#Top 50 negative tokens bar chart
y_pos = np.arange(50)
plt.figure(figsize=(20,10))
plt.bar(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 negative tokens')
plt.title('Top 50 tokens in negative tweets')


#Top 50 positive tokens bar chart
y_pos = np.arange(50)
plt.figure(figsize=(20,10))
plt.bar(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 positive tokens')
plt.title('Top 50 tokens in positive tweets')


# In[ ]:


group_by_month_historical = df.groupby(["Month"]).size().to_frame('Count').reset_index().sort_values(by=['Count'],ascending=False)
group_by_month_historical['Month'] = pd.Categorical(group_by_month_historical['Month'], ["January", "February", "March","April","May","June","July","August","September","October","November","December"])
group_by_month_historical = group_by_month_historical.sort_values(by=["Month"])

print("Historical (2013-2014)")
print(group_by_month_historical)


# In[ ]:


group_by_month_presidential = df_current.groupby(["Month"]).size().to_frame('Count').reset_index().sort_values(by=['Count'],ascending=False)
group_by_month_presidential['Month'] = pd.Categorical(group_by_month_presidential['Month'], ["January", "February", "March","April","May","June","July","August","September","October","November","December"])
group_by_month_presidential = group_by_month_presidential.sort_values(by=["Month"])

print("Presidential (2015-2018)")
print(group_by_month_presidential)


# In[ ]:


objects_historical = tuple(group_by_month_historical["Month"].tolist())
y_pos_historical = np.arange(len(objects_historical))
performance_historical = group_by_month_historical["Count"].tolist()
performance_presidential = group_by_month_presidential["Count"].tolist()

plt.figure(figsize=(20,5))
p1 = plt.bar(y_pos_historical - 0.2, performance_historical, align='center', alpha=0.5,width=0.4,color='g')
p2 = plt.bar(y_pos_historical + 0.2, performance_presidential, align='center', alpha=0.5,width=0.4,color='b')
plt.xticks(y_pos_historical, objects_historical)
plt.ylabel('Count')
plt.title('Number of Tweets per month\n (2013-2014) and (2015-2018)')
plt.legend((p1[0], p2[0]), ('Historical (2013 - 2014)','Current (2015 - 2018)'))
plt.show()


# In[ ]:


group_by_day = df.groupby(["Day_of_the_Week"]).size().to_frame('Count').reset_index().sort_values(by=['Count'],ascending=False)
group_by_day['Day_of_the_Week'] = pd.Categorical(group_by_day['Day_of_the_Week'], ["Sunday", "Monday", "Tuesday","Wednesday","Thursday","Friday","Saturday"])
group_by_day = group_by_day.sort_values(by=["Day_of_the_Week"])
objects = tuple(group_by_day["Day_of_the_Week"].tolist())
y_pos = np.arange(len(objects))
performance = group_by_day["Count"].tolist()

group_by_day_current = df_current.groupby(["Day_of_the_Week"]).size().to_frame('Count').reset_index().sort_values(by=['Count'],ascending=False)
group_by_day_current['Day_of_the_Week'] = pd.Categorical(group_by_day_current['Day_of_the_Week'], ["Sunday", "Monday", "Tuesday","Wednesday","Thursday","Friday","Saturday"])
group_by_day_current = group_by_day_current.sort_values(by=["Day_of_the_Week"])
performance_current = group_by_day_current["Count"].tolist()

plt.figure(figsize=(20,5))
p1 = plt.bar(y_pos - 0.2, performance, align='center', alpha=0.5,width=0.4,color='g')
p2 = plt.bar(y_pos + 0.2, performance_current, align='center', alpha=0.5,width=0.4,color='b')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Number of tweets based on the different days of the week\n (2013-2014) and (2015-2018)') 
plt.legend((p1[0], p2[0]), ('Historical (2013 - 2014)','Current (2015 - 2018)'))
plt.show()


# In[ ]:


# Number of tweets based on the hour, grouped by source\n Historical (2013-2014)

markers = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd']
# Formatting data to only use the hour it was sent, discard minutes and seconds
df["Hour"] = df["created_at"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%H'))
df_current["Hour"] = df_current["created_at"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%H'))
group_by_source = df[['source','Hour']].groupby(['source','Hour']).size().to_frame('Count').reset_index()

plt.figure(figsize=(20,10))
index = 0
for name, group in group_by_source.groupby('source'): 
    if(sum(i > 10 for i in group['Count'].tolist()) > 0):
        plt.plot(group['Hour'].apply(pd.to_numeric).tolist(),group['Count'].tolist(),markers[index]+'-', label=name)
        index += 1


plt.axis([0,24,group_by_source['Count'].min(),group_by_source['Count'].max()])
plt.ylabel('Count')
plt.xlabel('Hour of the Day')
plt.title('Number of tweets based on the hour, grouped by source\n Historical (2013-2014)') 
plt.legend()
plt.yscale('log')
plt.xticks(np.arange(0, 24, step=1))
plt.show()


# In[ ]:


group_by_source_current = df_current[['source','Hour']].groupby(['source','Hour']).size().to_frame('Count').reset_index()

plt.figure(figsize=(20,10))
index = 0
for name, group in group_by_source_current.groupby('source'): 
    if(sum(i > 10 for i in group['Count'].tolist()) > 0):
        plt.plot(group['Hour'].apply(pd.to_numeric).tolist(),group['Count'].tolist(),markers[index]+'-', label=name)
        index += 1

# print(group_by_source_current.describe())
plt.axis([0,24,group_by_source_current['Count'].min(),group_by_source_current['Count'].max()])
plt.ylabel('Count')
plt.xlabel('Hour of the Day')
plt.title('Number of tweets based on the hour, grouped by source\n Presidential (2015-2018)') 
plt.legend()
plt.yscale('log')
plt.show()


# In[ ]:


group_by_source_retweet_historical = df[['source','retweet_count']].groupby(['source']).size().to_frame('Retweet Count').reset_index()
print("Grouped by source and re-tweet count\n")
print("Historical")
print(group_by_source_retweet_historical)
group_by_source_retweet_presidential = df_current[['source','retweet_count']].groupby(['source']).size().to_frame('Retweet Count').reset_index()
print("\nPresidential")
print(group_by_source_retweet_presidential)


# In[ ]:


performance_historical = group_by_source_retweet_historical["Retweet Count"].tolist()
performance_current = group_by_source_retweet_presidential["Retweet Count"].tolist()

objects = tuple(group_by_source_retweet_historical["source"].tolist())
y_pos = np.arange(len(objects))

plt.figure(figsize=(30,10))
p1 = plt.bar(y_pos, performance_historical, align='center', alpha=0.5,width=0.4,color='g')
plt.xticks(y_pos, objects)
plt.ylabel('Retweet Count')
plt.title('Number of tweets based on \'retweet\' count grouped by source\nHistorical (2013-2014)') 
plt.yscale('log')
plt.show()

objects = tuple(group_by_source_retweet_presidential["source"].tolist())
y_pos = np.arange(len(objects))

plt.figure(figsize=(30,10))
p1 = plt.bar(y_pos, performance_current, align='center', alpha=0.5,width=0.4,color='b')
plt.xticks(y_pos, objects)
plt.ylabel('Retweet Count')
plt.title('Number of tweets based on \'retweet\' count grouped by source\n Presidential (2015-2018)') 
plt.yscale('log')
plt.show()



# In[ ]:


group_by_source_favorite_historical = df[['source','favorite_count']].groupby(['source']).size().to_frame('Favorite Count').reset_index()
print("Grouped by source and favorite count\n")
print("Historical")
print(group_by_source_favorite_historical)
group_by_source_favorite_presidential = df_current[['source','favorite_count']].groupby(['source']).size().to_frame('Favorite Count').reset_index()
print("\nPresidential")
print(group_by_source_favorite_presidential)


# In[ ]:


performance_historical = group_by_source_favorite_historical["Favorite Count"].tolist()
performance_current = group_by_source_favorite_presidential["Favorite Count"].tolist()

objects = tuple(group_by_source_favorite_historical["source"].tolist())
y_pos = np.arange(len(objects))

plt.figure(figsize=(30,10))
p1 = plt.bar(y_pos, performance_historical, align='center', alpha=0.5,width=0.4,color='g')
plt.xticks(y_pos, objects)
plt.ylabel('Favorite Count')
plt.title('Number of tweets based on \'favorite\' count grouped by source\nHistorical (2013-2014)') 
plt.yscale('log')
plt.show()

objects = tuple(group_by_source_favorite_presidential["source"].tolist())
y_pos = np.arange(len(objects))

plt.figure(figsize=(30,10))
p1 = plt.bar(y_pos, performance_current, align='center', alpha=0.5,width=0.4,color='b')
plt.xticks(y_pos, objects)
plt.ylabel('Favorite Count')
plt.title('Number of tweets based on \'favorite\' count grouped by source\n Presidential (2015-2018)') 
plt.yscale('log')
plt.show()


# In[ ]:


# Grouping by hour of the day


# In[ ]:


# Idea: Graph number of tweets per hour of the day
print("Number of tweets by time of day")
group_by_hour = df.groupby(['Hour']).size().to_frame('Count').reset_index()
group_by_hour_current = df_current.groupby(['Hour']).size().to_frame('Count').reset_index()
# print(group_by_hour)


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(group_by_hour['Hour'].tolist(),group_by_hour['Count'].tolist(),color='r',label="Historical")
plt.plot(group_by_hour_current['Hour'].tolist(),group_by_hour_current['Count'].tolist(),color='g',label="Current")

plt.ylabel('Count')
plt.xlabel('Hour of the Day')
plt.title('Number of tweets based on the hour') 
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:


# Idea: Graph quoted tweets by source
print("Number of quoted tweets by hour of the day")
df.groupby(['Hour','Quoted']).size().to_frame('Count').reset_index()
df_current.groupby(['Hour','Quoted']).size().to_frame('Count').reset_index()
group_quoted_by_source = df.groupby(['Hour','Quoted']).size().to_frame('Count').reset_index()
group_quoted_by_source_current = df_current.groupby(['Hour','Quoted']).size().to_frame('Count').reset_index()


# In[ ]:


# print(group_quoted_by_source)
plt.figure(figsize=(20,10))
plt.plot(group_quoted_by_source[group_quoted_by_source['Quoted'] == 0]['Hour'].tolist(),group_quoted_by_source[group_quoted_by_source['Quoted'] == 0]['Count'].tolist(),'ro-',color='r',label="Historical - ! Quoted")
plt.plot(group_quoted_by_source[group_quoted_by_source['Quoted'] == 1]['Hour'].tolist(),group_quoted_by_source[group_quoted_by_source['Quoted'] == 1]['Count'].tolist(),'r^-',color='r',label="Historical - Quoted")
plt.ylabel('Count')

plt.plot(group_quoted_by_source_current[group_quoted_by_source_current['Quoted'] == 0]['Hour'].tolist(),group_quoted_by_source_current[group_quoted_by_source_current['Quoted'] == 0]['Count'].tolist(),'ro-',color='b',label="Current - ! Quoted")
plt.plot(group_quoted_by_source_current[group_quoted_by_source_current['Quoted'] == 1]['Hour'].tolist(),group_quoted_by_source_current[group_quoted_by_source_current['Quoted'] == 1]['Count'].tolist(),'r^-',color='b',label="Current - Quoted")

plt.xlabel('Hour of the Day')
plt.title('Number of quoted tweets based on the hour of the day') 
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:


# Idea: Graph containing media tweets by hour of the day 
print("Number of tweets containing media by hour of the day")
group_media_by_hour = df.groupby(['Hour','Media']).size().to_frame('Count').reset_index()
group_media_by_hour_current = df_current.groupby(['Hour','Media']).size().to_frame('Count').reset_index()


# In[ ]:


# print(group_quoted_by_source)
plt.figure(figsize=(20,10))
plt.plot(group_media_by_hour[group_media_by_hour['Media'] == 0]['Hour'].tolist(),group_media_by_hour[group_media_by_hour['Media'] == 0]['Count'].tolist(),'ro-',color='r',label="Historical - ! Media")
plt.plot(group_media_by_hour[group_media_by_hour['Media'] == 1]['Hour'].tolist(),group_media_by_hour[group_media_by_hour['Media'] == 1]['Count'].tolist(),'r^-',color='r',label="Historical - Media")
plt.ylabel('Count')

plt.plot(group_media_by_hour_current[group_media_by_hour_current['Media'] == 0]['Hour'].tolist(),group_media_by_hour_current[group_media_by_hour_current['Media'] == 0]['Count'].tolist(),'ro-',color='b',label="Current - ! Media")
plt.plot(group_media_by_hour_current[group_media_by_hour_current['Media'] == 1]['Hour'].tolist(),group_media_by_hour_current[group_media_by_hour_current['Media'] == 1]['Count'].tolist(),'r^-',color='b',label="Current - Media")

plt.xlabel('Hour of the Day')
plt.title('Number of tweets containing media based on the hour of the day') 
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:


#Idea: Replicate media graph but using bars with AM/PM
print("Number of tweets containing media by AM/PM")
group_media_by_ampm = df.groupby(['AM/PM','Media']).size().to_frame('Count').reset_index()
group_media_by_ampm_current = df_current.groupby(['AM/PM','Media']).size().to_frame('Count').reset_index()


# In[ ]:


print(group_media_by_ampm)
print(group_media_by_ampm[group_media_by_ampm['Media'] == 0].groupby(['AM/PM']).sum()['Count'].tolist())


# In[ ]:


objects = tuple(group_media_by_ampm[group_media_by_ampm['Media'] == 0]["AM/PM"].tolist())
y_pos = np.arange(len(objects))
print(y_pos)

plt.figure(figsize=(10,10))
p1 = plt.bar(y_pos - 0.25,group_media_by_ampm[group_media_by_ampm['Media'] == 0].groupby(['AM/PM']).sum()['Count'].tolist(), align='center', alpha=0.5,width=0.5,color='b')

p2 = plt.bar(y_pos + 0.25,group_media_by_ampm[group_media_by_ampm['Media'] == 1].groupby(['AM/PM']).sum()['Count'].tolist(), align='center', alpha=0.5,width=0.5,color='g')
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Number of tweets containing media grouped by AM/PM') 
# plt.yscale('log')
plt.legend((p1[0], p2[0]), ('! Media', 'Media'))
plt.show()


# In[ ]:


# Idea: Graph SA per hour of the day
print("SA tweets by hour of the day")
group_sa_by_source = df.groupby(['SA','Hour']).size().to_frame('Count').reset_index()
group_sa_by_source_current = df_current.groupby(['SA','Hour']).size().to_frame('Count').reset_index()


plt.figure(figsize=(20,10))
plt.plot(group_sa_by_source[group_sa_by_source['SA'] == 0]['Hour'].tolist(),group_sa_by_source[group_sa_by_source['SA'] == 0]['Count'].tolist(),'-^',color='r',label="Historical - Neutral")
plt.plot(group_sa_by_source[group_sa_by_source['SA'] == 1]['Hour'].tolist(),group_sa_by_source[group_sa_by_source['SA'] == 1]['Count'].tolist(),'-o',color='r',label="Historical - Positive")
plt.plot(group_sa_by_source[group_sa_by_source['SA'] == -1]['Hour'].tolist(),group_sa_by_source[group_sa_by_source['SA'] == -1]['Count'].tolist(),'--',color='r',label="Historical - Negative")
plt.ylabel('Count')

plt.plot(group_sa_by_source_current[group_sa_by_source_current['SA'] == 0]['Hour'].tolist(),group_sa_by_source_current[group_sa_by_source_current['SA'] == 0]['Count'].tolist(),'-^',color='b',label="Current - Neutral")
plt.plot(group_sa_by_source_current[group_sa_by_source_current['SA'] == 1]['Hour'].tolist(),group_sa_by_source_current[group_sa_by_source_current['SA'] == 1]['Count'].tolist(),'-o',color='b',label="Current - Positive")
plt.plot(group_sa_by_source_current[group_sa_by_source_current['SA'] == -1]['Hour'].tolist(),group_sa_by_source_current[group_sa_by_source_current['SA'] == -1]['Count'].tolist(),'--',color='b',label="Current - Negative")

plt.xlabel('Hour of the Day')
plt.title('Number of tweets classified by SA on the hour of the day') 
plt.legend()
plt.grid(True)

plt.show()

