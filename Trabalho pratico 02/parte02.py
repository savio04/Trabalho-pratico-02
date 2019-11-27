from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler  


wini = datasets.load_wine() # Carregando o conjunto de dados

# Algoritimos utilizados 
modelo1 = svm.SVC(gamma=0.001,C=100)
modelo2 = ExtraTreesClassifier(n_estimators=100)
modelo3 = KNeighborsClassifier(n_neighbors=9)
modelo4 = GaussianNB()
escala = StandardScaler()

x = wini.data
y = wini.target

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size = 0.20,random_state =0) # Dividindo os dados em treino e teste


escala.fit(x_treino)
x_treino = escala.transform(x_treino)
x_teste = escala.transform(x_teste)

#Treinando e prevendo resultados do primeiro modelo1

modelo1.fit(x_treino,y_treino)
previsao1 = modelo1.predict(x_teste)
print("Classificação do svm: \n" , modelo1,'\n',classification_report(y_teste, previsao1))
print('Score: ', modelo1.score(x_teste,y_teste))

#Treinando e prevendo resultados do primeiro modelo2

modelo2.fit(x_treino,y_treino)
previsao2 = modelo2.predict(x_teste)
print('Classificação do ExtraTreesClassifier:\n',modelo2,'\n',classification_report(y_teste,previsao2))
print('Score: ', modelo2.score(x_teste,y_teste))
#Treinando e prevendo resultados do primeiro modelo2

modelo3.fit(x_treino,y_treino)
previsao3 = modelo3.predict(x_teste)
print('Classificação do KNeighborsClassifier:\n ',modelo3,'\n',classification_report(y_teste,previsao3))
print('Score: ', modelo3.score(x_teste,y_teste))
#Treinando e prevendo resultados do primeiro modelo2

modelo4.fit(x_treino,y_treino)
previsao4 = modelo4.predict(x_teste)
print('Classificação do GaussianNB :\n ',modelo4,'\n',classification_report(y_teste,previsao4))
print('Score: ', modelo4.score(x_teste,y_teste))
