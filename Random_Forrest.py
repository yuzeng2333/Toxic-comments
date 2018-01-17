import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import log_loss


import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')
subm = pd.read_csv('../../input/sample_submission.csv')

df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow_train = train.shape[0]


vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=50000, token_pattern=r'\w{1,}')
data = vectorizer.fit_transform(df)

X = MaxAbsScaler().fit_transform(data)

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((test.shape[0], len(col)))
num={}
loss = []
loss_max = 0;
losses=np.zeros((train.shape[0],1))


for idx, category in enumerate(col):
    print('=== Fit '+category)
    num[category] = train[category].sum()
    rf = RandomForestClassifier(n_estimators=600, n_jobs=-1, max_features=600, min_impurity_decrease=0.00001, bootstrap=True)
    rf.fit(X[:nrow_train], train[category])
    preds[:, idx] = rf.predict_proba(X[nrow_train:])[:,1]
    pred_train = rf.predict_proba(X[:nrow_train])[:,1]
    logloss = log_loss(train[category], pred_train)
    print('log loss: ', logloss)
    print('Avg loss: ', logloss/num[category])
    loss.append(log_loss(train[category], pred_train))

print('mean column-wise log loss:', np.mean(loss))

submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission_Random_Forrest_600_features.csv', index=False)
