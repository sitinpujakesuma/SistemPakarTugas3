import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
coba = pd.read_excel("data testing.xlsx")
coba.head(11)

coba.info()

from sklearn.cluster import KMeans

plt.scatter(coba.Matematika, coba.Jurusan, s = 75, c = "c", marker = "o", alpha = 0.5)
plt.show()

x = coba.drop(["Jurusan"], axis=1)
x.head(11)

y = coba["Jurusan"]
y.head(11)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

modelnb = GaussianNB()

nbtrain = modelnb.fit(x, y)
uji = pd.read_excel("data uji akurasi.xlsx")
uji.head(11)

x_test = uji.drop(["Jurusan"], axis=1)
x_test.head(11)

y_uji = uji["Jurusan"]
y_uji.head(11)

Y_predict = nbtrain.predict(x_test)
print("Prediksi Naive Bayes : ",Y_predict)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_uji, Y_predict)
print("Akurasi Naive Bayes : ",accuracy)
 
from sklearn.metrics import classification_report
print(classification_report(y_uji, Y_predict))
