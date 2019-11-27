from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

digitos = datasets.load_digits() # carregando o conjunto de dados

images_and_labels = list(zip(digitos.images, digitos.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Treino: %i' % label)


#transformando os dados em uma matriz
n_samples = len(digitos.images)
data = digitos.images.reshape((n_samples, -1))



#Classificação usando svm
modelo0 = svm.SVC(gamma= 0.001,C=100)
modelo0.fit(data[:n_samples//2], digitos.target[:n_samples//2]) #treinando o modelo0

y0_teste = digitos.target[n_samples//2:]
y_pred = modelo0.predict(data[n_samples//2:]) #prevendo resultados, a partir de valores que o modelo nao viu ainda 

#Algoritimo adicionado usando o ExtraTreesClassifier
modelo = ExtraTreesClassifier(n_estimators=100)

x= digitos.data # x armazena todos os dados que são analisados pelo o modelo e a partir dessa analise é feita a previsão
y = digitos.target # y armazena os resultados do conjunto de dados


x_treino,x_teste,y_treino,y_teste = train_test_split(x,y,test_size=0.20,random_state =0) #dividindo os dados para treino e teste
modelo.fit(x_treino,y_treino)# treinando o modelo 

previsao = modelo.predict(x_teste)#prevendo resultados, a partir de valores que o modelo nao viu ainda 

#Algoritimo adicionado usando o KNeighborsClassifier
modelo1 = KNeighborsClassifier(n_neighbors=5)
modelo1.fit(x_treino,y_treino)#Treinando o modelo1

previsao1 = modelo1.predict(x_teste)#prevendo os resultados, a partir de valores que o modelo nao viu ainda 


print('Classificação do svm: ',modelo0, '\n', classification_report(y0_teste,y_pred))
print('Score:', modelo0.score(data[n_samples//2:],y0_teste))
print('Classificação do ExtraTreesClassifier: ', modelo, '\n', classification_report(y_teste,previsao))
print('Score:', modelo.score(x_teste,y_teste))
print('Classificação do KNeighborsClassifier: ', modelo1, '\n', classification_report(y_teste,previsao1))
print('Score:', modelo1.score(x_teste,y_teste))
images_and_predictions = list(zip(digitos.images[n_samples // 2:], y_pred))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4 , index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Previsão: %i' % prediction)

plt.show()