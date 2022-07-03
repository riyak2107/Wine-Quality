#Wine Quality Prediction

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

lr=LogisticRegression(solver='sag',max_iter=10000)
rf=RandomForestClassifier(random_state=1)
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nn=MLPClassifier(solver='sgd', max_iter=10000, alpha=1e-4, hidden_layer_sizes=(1,1), random_state=0)
nb=MultinomialNB()
gb=GaussianNB()

df=pd.read_csv("C:/Users/Riya/Downloads/winequalityN.csv")

#print(df)

#print(x)
#print(y)

#df.info()

#print(df.isna().sum())

df['fixed acidity'] = df['fixed acidity'].fillna(df['fixed acidity'].median())
df['volatile acidity'] = df['volatile acidity'].fillna(df['volatile acidity'].median())
df['citric acid'] = df['citric acid'].fillna(df['citric acid'].median())
df['residual sugar'] = df['residual sugar'].fillna(df['residual sugar'].median())
df['chlorides'] = df['chlorides'].fillna(df['chlorides'].median())
df['pH'] = df['pH'].fillna(df['pH'].median())
df['sulphates'] = df['sulphates'].fillna(df['sulphates'].median())

# df["fixed acidity"].fillna(0, inplace = True)
# df["volatile acidity"].fillna(0, inplace = True)
# df["citric acid"].fillna(0, inplace = True)
# df["residual sugar"].fillna(0, inplace = True)
# df["chlorides"].fillna(0, inplace = True)
# df["pH"].fillna(0, inplace = True)
# df["sulphates"].fillna(0, inplace = True)

x1=df.drop("type",axis=1)
x=x1.drop("quality",axis=1)
y=df["quality"]

#print(df.isna().sum())
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)

#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)


# lr.fit(x_train,y_train)
# y_lrp=lr.predict(x_test)

rf.fit(x_train,y_train)
y_rfp=rf.predict(x_test)

# gbm.fit(x_train,y_train)
# y_gbmp=gbm.predict(x_test)
#
# dt.fit(x_train,y_train)
# y_dtp=dt.predict(x_test)
#
# sv.fit(x_train,y_train)
# y_svp=sv.predict(x_test)
#
# nn.fit(x_train,y_train)
# y_nnp=nn.predict(x_test)
#
# nb.fit(x_train,y_train)
# y_nbp=nb.predict(x_test)
#
# gb.fit(x_train,y_train)
# y_gbp=gb.predict(x_test)

#print('Logistic Regression : ' ,accuracy_score(y_test,y_lrp))
print('Random forest : ' ,accuracy_score(y_test,y_rfp)*100)
# print('Gradient Boosting Method : ' ,accuracy_score(y_test,y_gbmp))
# print('Decision Tree : ' ,accuracy_score(y_test,y_dtp))
# print('SVM : ' ,accuracy_score(y_test,y_svp))
# print('Neural Networks : ' ,accuracy_score(y_test,y_nnp))
# print('Naive Bayes : ' ,accuracy_score(y_test,y_nbp))
# print('GaussianNB : ' ,accuracy_score(y_test,y_gbp))

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# bestfeaures = SelectKBest(score_func=chi2, k='all')
# fit = bestfeaures.fit(x,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(x.columns)
# featuresScores = pd.concat([dfcolumns,dfscores],axis=1)
# featuresScores.columns=['Specs','Score']
#
# print(featuresScores)

from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x,y=ros.fit_resample(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
rf.fit(x_train,y_train)
y_rfp=rf.predict(x_test)

print('After feature engineering Random forest : ' ,accuracy_score(y_test,y_rfp)*100)

'''
Output :  
Random forest :  68.38461538461539
After feature engineering Random forest :  93.4273482749937

Process finished with exit code 0
'''
