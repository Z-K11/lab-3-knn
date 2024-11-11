import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#start from here

readable  = pd.read_csv('teleCust1000t.csv')
# reading the data from csv file storing it to our variable
# labels within csv region,tenure,age,marital,address,income,ed,employ,retire,gender,reside,custcat
#print(readable['income'].value_counts())
# visual Representation
#readable.hist(column='income',bins=50)
#plt.savefig("income.png")
#print(readable.columns)
X = readable[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']].values.astype(float)
Y = readable['custcat'].values.astype(float)
scaler = preprocessing.StandardScaler()
# Initialziing the scaller
X = scaler.fit_transform(X)
# Scalling x
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)
k = 16 
# k = 4 sets the bar to compare from 4 nearest neighbors 
model = KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
y_hat = model.predict(x_test)
print("Train set accuracy ", metrics.accuracy_score(y_train,model.predict(x_train)))
print("Test set accuracy ", metrics.accuracy_score(y_test,y_hat))

Ks =20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    model2 = KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    prediction = model2.predict(x_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,prediction)
    std_acc[n-1]=np.std(prediction==y_test)/np.sqrt(prediction.shape[0])
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.savefig("result.png")
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
