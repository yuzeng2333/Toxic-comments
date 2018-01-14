# spammers should be fixed later
import pandas as pd 
import numpy as np

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy

#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD

path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE='../input/glove6b50d/glove.6B.50d.txt'
TRAIN_DATA_FILE='../input/train.csv'
TEST_DATA_FILE='../input/test.csv'
TRAIN_CLEAN_FILE='../input/train_clean.csv'
TEST_CLEAN_FILE='../input/test_clean.csv'


train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

# clean
merge=pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
corpus=merge.comment_text
df=merge.reset_index(drop=True)

from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
# !!! cannot import TweetTokenize
#from nltk.tokenize import TweetTokenize  
from nltk.tokenize import LineTokenizer  

tokenizer = LineTokenizer()
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
eng_stopwords = set(stopwords.words("english"))

# Leaky features should be addressed later
df['ip']=df["comment_text"].apply(lambda x: re.findall("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}",str(x)))
#count of ip addresses
df['count_ip']=df["ip"].apply(lambda x: len(x))

#links
df['link']=df["comment_text"].apply(lambda x: re.findall("http://.*com",str(x)))
#count of links
df['count_links']=df["link"].apply(lambda x: len(x))

#article ids
df['article_id']=df["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$",str(x)))
df['article_id_flag']=df.article_id.apply(lambda x: len(x))

#username
##              regex for     Match anything with [[User: ---------- ]]
# regexp = re.compile("\[\[User:(.*)\|")
df['username']=df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|",str(x)))
#count of username mentions
df['count_usernames']=df["username"].apply(lambda x: len(x))

# check if the input is float
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def clean(comment):
    #Convert to lower case , so that Hi and hi are the same
    if not isfloat(comment):
        comment=comment.lower()
    #remove \n
    #comment=re.sub("\\n","",str(comment))
    # remove the '\'
    #comment=re.sub("\\\'","'",str(comment))
    # remove leaky elements like ip, links
    #Split the sentences into words
    words=tokenizer.tokenize(comment)    
    # (')aphostophe  replacement (ie)   you're --> you are
    #words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    # remove stop words
    #words = [w for w in words if not w in eng_stopwords]    
    comment=" ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(comment)


def clean_all(comment):
    #Convert to lower case , so that Hi and hi are the same
    if not isfloat(comment):
        comment=comment.lower()
    #remove \n
    comment=re.sub("\\n","",str(comment))
    # remove the '\'
    comment=re.sub("\\\'","'",str(comment))
    # remove leaky elements like ip
    comment=re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","",str(comment))
    # remove links
    comment=re.sub("http://.*com","",str(comment))
    # remove artical ids
    comment=re.sub("\d:\d\d\s{0,5}$","",str(comment)) 
    # remove user name
    comment=re.sub("\[\[User(.*)\|","",str(comment)) 
         #removing usernames
    comment=re.sub("\[\[.*\]","",str(comment))    
    #Split the sentences into words
    words=tokenizer.tokenize(comment)    
    # (')aphostophe  replacement (ie)   you're --> you are
    #words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    # remove stop words
    #words = [w for w in words if not w in eng_stopwords]    
    clean_sent=" ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)

train.iloc[:,1] = train.iloc[:,1].apply(clean)
test.iloc[:,1] = test.iloc[:,1].apply(clean)

train.to_csv(path_or_buf=TRAIN_CLEAN_FILE)
test.to_csv(path_or_buf=TEST_CLEAN_FILE)

