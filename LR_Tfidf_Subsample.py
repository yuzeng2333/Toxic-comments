# in this version, several LR classifier will be used for a single class, and the final result is the average of all the classifiers.
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import gc

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')



df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow_train = train.shape[0]


vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=50000, token_pattern=r'\w{1,}')
data = vectorizer.fit_transform(df)

X = MaxAbsScaler().fit_transform(data)

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

preds = np.zeros((test.shape[0], len(col)))

# calculate the number of each class
num={}
#for 
#num['toxic'] = train['toxic']



loss = []
loss_max = 0;
losses=np.zeros((train.shape[0],1))

for i, j in enumerate(col):
    print('===Fit '+j)
    num[j] = train[j].sum()
    number = str(num[j])
    num_classifier = int(nrow_train / num[j])
    pos_idx = [i for i,x in enumerate(train[j]) if x==1]
    neg_idx = [i for i,x in enumerate(train[j]) if x==0]
    print('The # of'+j+' is '+number) 
    preds = np.zeros((test.shape[0], len(col)))
    pred_train = np.zeros((train.shape[0], 1))
    for k in range(0,num_classifier):
        print('Train the '+str(k)+' classifier')
        neg_rdm = np.random.choice(neg_idx, size = num[j])
        #train_sample = pd.concat([ X[pos_idx], X[neg_rdm] ], axis=0)
        X_idx = pos_idx + list(neg_rdm)
        train_sample = X[pos_idx+list(neg_rdm)]
        label_sample = pd.concat([ train[j][pos_idx], train[j][neg_rdm] ], axis=0)
        model = LogisticRegression(C=3)
        model.fit(train_sample, label_sample)
        preds[:,i] = ( np.squeeze(preds[:,i])*k + model.predict_proba(X[nrow_train:])[:,1] ) / (k+1)
        pred_train = ( np.squeeze(pred_train)*k + model.predict_proba(X[:nrow_train])[:,1] ) / (k+1)
        del model
        gc.collect()
        logloss = log_loss(train[j], pred_train)
        print(str(k)+': log loss:', logloss)
        print(str(k)+': Avg_loss:', logloss/num[j])
 
    loss.append(log_loss(train[j], pred_train))

    # calculate the log loss of each pair

print('mean column-wise log loss:', np.mean(loss))
    
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission.csv', index=False)
