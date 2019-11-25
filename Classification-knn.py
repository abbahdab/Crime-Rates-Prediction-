import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics 
import seaborn as sns 
import numpy as np

crimes = pd.read_csv("Crime_Data_2010_2017_clean.csv")
#print(crimes)
#print(crimes.head())
#print(crimes.shape)

''''Morning: 5am - 11:59am         0500 - 1159
    Afternoon: 12pm - 4:59pm       1200 - 1659
    Evening: 5:00pm - 8:59pm       1700 - 2059 
    Night: 9:00pm - 4:59am         2100 - 0459'''    

crimesTime = crimes['Time Occurred']
crimesTimeName = []
mornCount = 0
afterCount = 0
evenCount = 0
nightCount = 0
#set each time to morning, afternoon, evening or night 

for i in crimesTime:
    if i in range(500, 1159):
        crimesTimeName.append("Morning")
        mornCount += 1
    elif i in range(1200,1659):
        crimesTimeName.append("Afternoon")
        afterCount += 1
    elif i in range(1700, 2059):
        crimesTimeName.append("Evening")
        evenCount += 1
    else:
        crimesTimeName.append("Night")
        nightCount += 1
    
crimes['Time of Day'] = crimesTimeName

#print(len(crimes))
#print((crimesTime))

print('Crimes committed in the morning:',mornCount)
print('Crimes committed in the afternoon:',afterCount)
print('Crimes committed in the evening:',evenCount)
print('Crimes committed in the night:',nightCount)

sns.countplot(crimes['Time of Day'], label = 'Crime Count')
plt.show()

crimes = crimes.dropna()

target = crimes.drop(columns = ['Time of Day'])
target.head()

y = crimes['Time of Day']

X_train, X_test, y_train, y_test = train_test_split(target, y, test_size = 0.6)

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier with k = 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Import confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

k = 0
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(k,"Accuracy:",metrics.accuracy_score(y_test, y_pred))
