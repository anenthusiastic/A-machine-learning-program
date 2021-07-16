import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np 

#Çeşitli verileri verilen bir çifti, doğum kontrol yöntemi "kullanmıyorlar","uzun vadeli kullanıyorlar" veya  
#"kısa vadeli kullanıyorlar" şeklinde sınıflandıran program

""" Attribute Information:

   1. Wife's age                     (numerical)
   2. Wife's education               (categorical)      1=low, 2, 3, 4=high
   3. Husband's education            (categorical)      1=low, 2, 3, 4=high
   4. Number of children ever born   (numerical)
   5. Wife's religion                (binary)           0=Non-Islam, 1=Islam
   6. Wife's now working?            (binary)           0=Yes, 1=No
   7. Husband's occupation           (categorical)      1, 2, 3, 4
   8. Standard-of-living index       (categorical)      1=low, 2, 3, 4=high
   9. Media exposure                 (binary)           0=Good, 1=Not good
   10. Contraceptive method used     (class attribute)  1=No-use 
                                                        2=Long-term
                                                        3=Short-term

"""

att_col_names= ['wife_age', 'wife_edu', 'husband_edu', 'child', 'religion', 'work', 'occupation','living_standarts','media_exposure']
df=pd.read_csv('C://Users/fatih/Desktop/anket.csv',sep=',')
df=df[1:]
X=df[att_col_names]
y=df[['method_use']]
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7, test_size=0.3, random_state=0,stratify=y)
clf = DecisionTreeClassifier(random_state=0)
dt_cross_val=cross_val_score(clf, X, y, cv=10)
print("DecisionTree 10-Fold Cross-Validation Result: ",dt_cross_val)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n",cm)
plt.matshow(cm) 
plt.title('Confusion matrix') 
plt.colorbar() 
plt.ylabel('True label') 
plt.xlabel('Predicted label') 
plt.show()

clf2 = MLPClassifier()
y=np.array(y)
y.shape=(1471,)
mlp_cross_val=cross_val_score(clf2, X, y, cv=10)
print("MLPClassifier 10-Fold Cross-Validation Result: ",mlp_cross_val)
y_train=np.array(y_train)
y_train.shape=(1029,)
clf2.fit(X_train,y_train)
y_pred = clf2.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n",cm)
print("---------------------------")
print("10-Fold Cross-Validation Results Chart")
print(" "*20,end="")
for i in range(10):
    print(i+1,"       ",end="")
print("")
print("{:15}".format("DecisionTree"),end="")
for i in range(10):
    print("{:9.5f}".format(dt_cross_val[i]),end="")
print("")
print("{:15}".format("MLPClassifier"),end="")
for i in range(10):
    print("{:9.5f}".format(mlp_cross_val[i]),end="")
print("")
print("")

print("Görüldüğü gibi MLPClassifier daha doğru sonuçlar üretmektedir.")

print("")

print("Kullanıcının girdiği verilerle sınıflandırma yapma--------------")
yaş=int(input("Wife's age : "))
wedu=int(input("Wife's education (1=low, 2, 3, 4=high): "))
hedu=int(input("Husband's education (1=low, 2, 3, 4=high): "))
child=int(input("Number of children ever born: "))
religion=int(input("Wife's religion (0=Non-Islam, 1=Islam): "))
work=int(input("Wife's now working? (0=Yes, 1=No): "))
occ=int(input("Husband's occupation  (1, 2, 3, 4): "))
living=int(input("Standard-of-living index (1=low, 2, 3, 4=high): "))
media=int(input("Media exposure  (0=Good, 1=Not good): "))

kullanıcı_verisi=[[yaş,wedu,hedu,child,religion,work,occ,living,media]]
pred=clf2.predict(kullanıcı_verisi)
if(pred[0]==1):
    print("Doğum kontrol yöntemi kullanılmıyor.")

if(pred[0]==2):
    print("Uzun süreli doğum kontrol yöntemi kullanılıyor.")

if(pred[0]==3):
    print("Kısa süreli doğum kontrol yöntemi kullanılıyor.")