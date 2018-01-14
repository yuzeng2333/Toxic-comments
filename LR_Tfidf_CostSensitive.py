import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
#from sklearn.linear_model import LogisticRegression
from costcla.models import CostSensitiveLogisticRegression
from sklearn.metrics import log_loss

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
    print('The # of'+j+' is '+number)
    model = CostSensitiveLogisticRegression(C=3)
    num[j] = train[j].sum()
    fn = nrow_train / num[j]
    cost_mat=np.ones((nrow_train, 4))
    #cost_mat[:, 1] = fn
    model.fit(X[:nrow_train], train[j], cost_mat)
    preds[:,i] = model.predict_proba(X[nrow_train:])[:,1]
    pred_train = model.predict_proba(X[:nrow_train])[:,1]
    logloss = log_loss(train[j], pred_train)
    print('log loss:', logloss)
    print('Avg_loss:', logloss/num[j])
    loss.append(log_loss(train[j], pred_train))

    # calculate the log loss of each pair

    
print('mean column-wise log loss:', np.mean(loss))
    
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission.csv', index=False)
