from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import math, random, multiprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,precision_score,recall_score 
import numpy as np
from sklearn.decomposition import PCA
#se supone que aqui va el clasificador hehcho por mi...
def Perceptron(X,y,queue, e = 0.0, yi = 0.05, m = 1): #que se procesa en paralelo
  Pesos  = np.array([random.random() for i in range(len(X[0])+1)]) #genera pesos aleatoriamente
  Pesos = Pesos/math.sqrt(sum(Pesos**2)) #los normaliza
  for i in range(1, 1000):  
    n = 1/(4*math.sqrt(i))  
    for Vector,clase in zip(X,y):
        Vector = np.append(Vector, 1)
        Sol = sum(Vector*Pesos) 
        o = 1 if (Sol >= 0) else (-1) 
        bandera = False 
        if o > clase+e and Sol >=0:
            z = Vector*(-1)  
        elif o < clase-e and Sol < 0:
            z = Vector   
        elif o <= clase+e and (Sol >=0 and Sol < yi):
            z = m*Vector
        elif o >= clase-e and (Sol >=-yi and Sol < 0):
            z = m*(Vector*(-1)) 
        else :  
            bandera = True 
            z = 0 
        Pesos = Pesos+(n*z) 
        if bandera != True: 
            Pesos = Pesos/math.sqrt(sum(Pesos**2))
  return queue.put(Pesos)


class subdatos:
    def __init__(self, X,y):
        self.X = X
        self.y = y


class UnivAprox(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_parame='demo'):
        self.demo_param = demo_parame
    def fit(self, X, y):
         # Check that X and y have correct shape
         clas = []
         clas = [clas.append(c) for c in y if c not in clas]
         X, y = check_X_y(X, y)
         # Store the classes seen during fit
         self.classes_ = unique_labels(y)
         self.X_ = X
         self.y_ = y
         self.rdlr = []
         #genera una serie de subdatos
         sub = []
         for j in range(1,len(clas)):
             X_sub = np.array([v for v,c in zip(X,y) if c<=j and j >0])
             y_sub = np.array([1 if i==j else -1 for i in y if i<=j and j >0])
             #print(y_sub)
             queue = multiprocessing.Queue() # inicializa el método Queue para obtener el resultado de cada proceso  
             sub.append(subdatos(X_sub,y_sub))
         for conj in reversed(sub):
             Procesos = [multiprocessing.Process(target = Perceptron, args= (conj.X,conj.y,queue)) for i in range(4)] #genera la cantidad de procesos que se necesiten
             [Proceso.start() for Proceso in Procesos] #inicializa los procesos
             [Proceso.join() for Proceso in Procesos] #se espera a que todos terminen
             resultados = []
             while  not queue.empty(): #toma los vectores de pesos obtenidos y lo mete en un arreglo 
                resultados.append(queue.get())
             self.rdlr.append(resultados)
         return self
    def predict(self, X):
         check_is_fitted(self)
         X = check_array(X)
         res = []
         for out in self.rdlr:
             res.append([1 if sum([sum(np.append(ent,1)*vec) for vec in out]) >0 else 0 for ent in X])
         np.array(res)
         reg = []
         for i in range(len(X)):
             reg.append(sum([res[j][i] for j in range(len(res))]))
         return np.array(reg)

#datasets externos
def load_titanic():
    dat = open("titanic.txt")
    X = np.array([np.fromstring(i, dtype = float, sep = ',') for i in dat.readlines()])
    y = np.array([i[0]for i in X])
    X = X[:, 1:]
    return X,y
def load_creado():
    dat = open("creado.txt")
    X = np.array([np.fromstring(i, dtype = float, sep = ',') for i in dat.readlines()])
    y = np.array([i[0]for i in X])
    X = X[:, 1:]
    return X,y

if __name__ == '__main__':
    
    #uso de datasets
    X,y = load_iris(return_X_y=True)
    #X,y = load_creado()
    #X,y = load_digits(return_X_y=True)
    #X,y = load_titanic()


    #escala los padrones del dataset
    escalador = StandardScaler()
    #usa el algoritmo PCA 
    componentes = 0.9
    pca = PCA(n_components=componentes)
    X = escalador.fit_transform(X)
    X_pca = pca.fit_transform(X) 
    print(pca.explained_variance_ratio_)

    #recortar
    #clases = [0,1,2]
    #X = np.array([v for v,c in zip(X,y) if c in clases])
    #y = np.array([i for i in y if i in clases])

    
    #uso de clasificadores
    clf = svm.SVC(kernel='rbf')
    #clf = UnivAprox()
    #clf = KNeighborsClassifier(n_neighbors=3) 
    #pliegues
    K =10
    #clf.fit(X,y)
    #kfolds
    kfcvSF = KFold(n_splits=K)
    skfcvSF = StratifiedKFold(n_splits=K)
    kfcvST = KFold(n_splits=K,shuffle=True)
    skfcvST = StratifiedKFold(n_splits=K,shuffle=True)

    for train_index, test_index in kfcvST.split(X,y):
        #
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        accsp = accuracy_score(y[test_index], y_pred)
        res_sin_pca_prueba = precision_recall_fscore_support(y[test_index], y_pred, zero_division=1)
        y_pred = clf.predict(X[train_index])
        accse = accuracy_score(y[train_index], y_pred)
        res_sin_pca_entrena = precision_recall_fscore_support(y[train_index], y_pred ,zero_division=1)
        
        print("Sin PCA")
        print(f"Datos de prueba \n Accuracy :{accsp}, Precision :{sum(res_sin_pca_prueba[0])/len(res_sin_pca_prueba[0])},Recall : {sum(res_sin_pca_prueba[1])/len(res_sin_pca_prueba[1])},\nDatos de experimentacion \nAccuracy : {accse}, Precison{sum(res_sin_pca_entrena[0])/len(res_sin_pca_entrena[0])}, Recall : {sum(res_sin_pca_entrena[1])/len(res_sin_pca_entrena[1])}\n")

        clf.fit(X_pca[train_index], y[train_index])
        y_pred = clf.predict(X_pca[test_index])
        acccp = accuracy_score(y[test_index], y_pred)
        res_con_pca_prueba = precision_recall_fscore_support(y[test_index], y_pred, zero_division=1)
        y_pred = clf.predict(X_pca[train_index])
        accce = accuracy_score(y[train_index], y_pred)
        res_con_pca_entrena = precision_recall_fscore_support(y[train_index], y_pred, zero_division=1)
        print("Con PCA")
        print(f"Datos de prueba \n Accuracy : {acccp}, Precision :{sum(res_con_pca_prueba[0])/len(res_con_pca_prueba[0])}, recall : {sum(res_con_pca_prueba[1])/len(res_con_pca_prueba[1])},\nDatos de experimentacion \nAccuracy : {accce}, Precision : {sum(res_con_pca_entrena[0])/len(res_con_pca_entrena[0])},Recall : {sum(res_con_pca_entrena[1])/len(res_con_pca_entrena[1])}\n")
        