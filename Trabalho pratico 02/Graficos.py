import matplotlib.pyplot as plt

#Gráfico da parte 01
algo = ['Svm', ' ExtraTrees', 'KNeighbors']
score = [96,98,97]
xs = [i + 0.8 for i, _ in enumerate(algo)]
plt.bar(xs, score,color = 'c')

plt.title('Gráfico da Parte 01',size = 14)
plt.ylabel('Score (%)',size = 12)
plt.xlabel('Algoritmos Usados',size = 12)
plt.xticks([i + 0.8 for i, _ in enumerate(algo)], algo)
plt.show()

# Gráfico da parte 02
algo = ['Svm', ' ExtraTrees', 'KNeighbors','GaussianNB']
score = [100,97,100,91]
xs = [i + 0.8 for i, _ in enumerate(algo)]
plt.bar(xs, score,color = 'c')

plt.title('Gráfico da Parte 02',size = 14)
plt.ylabel('Score (%)',size = 12)
plt.xlabel('Algoritmos Usados',size = 12)
plt.xticks([i + 0.8 for i, _ in enumerate(algo)], algo)
plt.show()