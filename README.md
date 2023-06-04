# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: N.Kishore
RegisterNumber: 212222240049
*/

import chardet
file="/content/spam.csv"
with open(file,"rb") as rawdata:
  result=chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

![Screenshot 2023-06-04 163713](https://github.com/nkishore2210/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707090/ea85cbda-a90d-4096-8978-255af1441023)

![Screenshot 2023-06-04 163721](https://github.com/nkishore2210/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707090/e0f054dd-63ae-4d02-bece-edb45d4f5391)

![Screenshot 2023-06-04 163733](https://github.com/nkishore2210/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707090/a189c3bd-de7b-4223-ad37-0126f745ad64)

![Screenshot 2023-06-04 163743](https://github.com/nkishore2210/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707090/6172e181-1f98-40dc-b0c1-be1ecb3582fd)

![Screenshot 2023-06-04 163751](https://github.com/nkishore2210/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707090/7a3441cf-4f06-46c5-a0cf-1a285a6218b2)

![Screenshot 2023-06-04 163759](https://github.com/nkishore2210/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707090/43633fbd-36b2-40b2-a1db-a197b25d3b4d)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
