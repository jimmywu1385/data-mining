import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
##############preprocessing######################################
data=pd.read_csv('top50.csv',encoding='ISO-8859-1')
data['Popularity']=data['Popularity'].fillna(data['Popularity'].mean())
data['Genre']=['hip hop' if each =='atl hip hop'
                      else 'hip hop' if each =='canadian hip hop'
                      else 'hip hop' if each == 'trap music'
                      else 'pop' if each == 'australian pop'
                      else 'pop' if each == 'boy band'
                      else 'pop' if each == 'canadian pop'
                      else 'pop' if each == 'dance pop'
                      else 'pop' if each == 'panamanian pop'
                      else 'pop' if each == 'pop'
                      else 'pop' if each == 'pop house'
                      else 'electronic' if each == 'big room'
                      else 'electronic' if each == 'brostep'
                      else 'electronic' if each == 'edm'
                      else 'electronic' if each == 'electropop'
                      else 'rap' if each == 'country rap'
                      else 'rap' if each == 'dfw rap'
                      else 'escape room' if each == 'hip hop'
                      else 'latin' if each == 'latin'
                      else 'r&b' if each == 'r&n en espanol'
                      else 'raggae' for each in data['Genre']]
x=data.loc[:,['Beats.Per.Minute','Energy','Danceability','Loudness..dB..','Liveness',
                'Valence.','Length.','Acousticness..','Speechiness.','Popularity']]
y=data.loc[:,'Genre']
sc=StandardScaler()
sc.fit(x)
x=sc.transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
###################knn######################################
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(x_train,y_train)
kpredict = knn.predict(x_test)
print('KNN Acurracy',knn.score(x_test,y_test))
###################svm#####################################
clf = SVC()
clf.fit(x_train,y_train)
cpredict=clf.predict(x_test)
print('SVM Acurracy',clf.score(x_test,y_test))
##################NVBayes####################################
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gpredict=gnb.predict(x_test)
print('NVBayes Acurracy',gnb.score(x_test,y_test))
###################RDForest###################################
rdf = RandomForestClassifier(n_estimators=2)
rdf.fit(x_train, y_train)
rpredict=rdf.predict(x_test)
print('RDForest Acurracy',rdf.score(x_test,y_test))

print(classification_report(y_test, kpredict))
print(classification_report(y_test, cpredict))
print(classification_report(y_test, gpredict))
print(classification_report(y_test, rpredict))


