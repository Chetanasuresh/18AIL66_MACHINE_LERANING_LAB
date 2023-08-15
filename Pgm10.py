import numpy as np
import matplotlib.puplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score , classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X,y=load_breast_cancer(return_X_y=True)
X_train, X-test, y_train, y_test=train_test_split(X[:,:2], y, test_size=0.2, random_state=42)
svm=SVC(kernel='linear',c=1)
svm.fit(X-trsin,y_train)
y_pred=svm.predict(X_test)
print('Accuracy:',acuracy_score(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))

from mlxtend.plotting import plot_decision_regions 
plot_decision_regions(X_train,y_train,clf=svm,legend=2) 
plt.xlabel('mean radius') 
plt.ylabel('mean texture') 
plt.title("SVM boundary decision") 
plt.show()
