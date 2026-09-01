import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

X,y=make_classification(300,n_features=2,n_informative=2,
n_redundant=0,n_classes=3,n_clusters_per_class=1,random_state=1)

m=MLPClassifier(hidden_layer_sizes=(8,),activation='tanh',
max_iter=1000).fit(X,y)

print("Accuracy:",m.score(X,y))
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
