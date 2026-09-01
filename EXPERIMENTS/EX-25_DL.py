import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

X,y=make_circles(300,noise=.1,factor=.5,random_state=1)
m=MLPClassifier(hidden_layer_sizes=(5,),activation='tanh',max_iter=1000).fit(X,y)
print("Accuracy:",m.score(X,y))
plt.scatter(X[:,0],X[:,1],c=y); plt.show()
