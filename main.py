import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

