from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def train_model(x_train, y_train):
    neighbors = [x for x in list(range(1,100)) if x % 2==0]
    cv_scores = []
    seed = 123
    for k in neighbors:
        k_value = k+1
        knn = KNeighborsClassifier(n_neighbors = k_value,weights='uniform',p=2,metric='euclidean')
        kfold = cross_val_score(knn,x_train,y_train,cv=10)
        cv_scores.append(kfold.mean()*100)
    optimal_k = neighbors[cv_scores.index(max(cv_scores))]
    knn = KNeighborsClassifier(n_neighbors = optimal_k)
    knn.fit(x_train, y_train)
    return knn