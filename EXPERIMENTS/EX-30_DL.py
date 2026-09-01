import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

X,y=make_classification(300,n_features=2,n_informative=2,
n_redundant=0,n_classes=2,random_state=1)

m=MLPClassifier(hidden_layer_sizes=(3,),activation='tanh',
max_iter=1000,random_state=1).fit(X,y)

print("Accuracy:",m.score(X,y))

plt.scatter(X[:,0],X[:,1],c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Two-Class Data - Tanh Activation")
plt.show()
