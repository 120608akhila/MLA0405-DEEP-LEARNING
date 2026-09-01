import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

t=np.linspace(0,4*np.pi,300)
X=np.c_[t*np.cos(t),t*np.sin(t)]
y=(t>2*np.pi).astype(int)
m=MLPClassifier(hidden_layer_sizes=(10,),activation='logistic',max_iter=2000).fit(X,y)
print("Accuracy:",m.score(X,y))
plt.scatter(X[:,0],X[:,1],c=y); plt.show()
