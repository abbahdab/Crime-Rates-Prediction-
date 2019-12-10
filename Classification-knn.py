import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics 
import seaborn as sns 
import numpy as np
import TransformScript as ts

crimes = pd.DataFrame()
crimes = ts.run(crimes, 'data/Crime_Data_2010_2017.csv')
#print(crimes)
#print(crimes.head())
#print(crimes.shape)

crimes = crimes[:len(crimes)//100]


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
    elif i in range(2100, 2400) or i in range(0, 459):
        crimesTimeName.append("Night")
        nightCount += 1
    else:
        crimesTimeName.append("NaN")

crimes['Time of Day'] = crimesTimeName

#print(len(crimes))
#print((crimesTime))

print('Crimes committed in the morning:',mornCount)
print('Crimes committed in the afternoon:',afterCount)
print('Crimes committed in the evening:',evenCount)
print('Crimes committed in the night:',nightCount)

sns.countplot(crimes['Time of Day'], label = 'count')
plt.xlabel('Time of Day')
plt.ylabel('Crime Count')
plt.title("Time of Day Crimes Occurred")
plt.show()

#target = crimes.dropna()

target = crimes.drop(columns = ['Time of Day','Time Occurred'])
target.head()

y = crimesTimeName

X_train, X_test, y_train, y_test = train_test_split(target, y, test_size = 0.2, random_state=0)
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
print(classification_report(y_test, y_pred))

k = 0
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(k,"Accuracy:",metrics.accuracy_score(y_test, y_pred))


crimes2 = pd.read_csv('data/Crime_Data_2010_2017.csv')
crimes2 = crimes2[:len(crimes2)//100]
weapons = crimes2['Weapon Used Code']

nonViolentCount = 0
violent = []
violentCount = 0

for i in weapons:
    if i in range(100, 1000):
        violentCount += 1
        violent.append("Violent")
    else:
        nonViolentCount += 1
        violent.append("Non-Violent")

crimes['Violence'] = violent
del crimes2

violent = crimes[crimes['Violence'] == "Violent"]
nonviolent = crimes[crimes['Violence'] == "Non-violent"]
sns.countplot(x = "Time of Day", color = "green", data = crimes)
sns.countplot(x = "Time of Day", hue = "Violence", palette = {"Violent": "#FF0000", "Non-Violent": "#3333FF"}, data = crimes)
plt.ylabel("Crime Count")
plt.title("Time of Day Violent and Non-Violent Crimes Occurred")
plt.show()
