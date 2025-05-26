import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Pré-processa imagens da pasta Treino (caminho embutido)
def preprocessar_imagem_treino(nome_arquivo):
    imagem = cv2.imread(f'Treino/{nome_arquivo}')
    imagem_redimensionada = cv2.resize(imagem, (128, 128))
    imagem_suavizada = cv2.GaussianBlur(imagem_redimensionada, (5, 5), 0)
    imagem_cinza = cv2.cvtColor(imagem_suavizada, cv2.COLOR_BGR2GRAY)
    imagem_equalizada = cv2.equalizeHist(imagem_cinza)
    return imagem_equalizada

# Pré-processa imagens da pasta Teste (caminho embutido)
def preprocessar_imagem_teste(nome_arquivo):
    imagem = cv2.imread(f'Teste/{nome_arquivo}')
    imagem_redimensionada = cv2.resize(imagem, (128, 128))
    imagem_suavizada = cv2.GaussianBlur(imagem_redimensionada, (5, 5), 0)
    imagem_cinza = cv2.cvtColor(imagem_suavizada, cv2.COLOR_BGR2GRAY)
    imagem_equalizada = cv2.equalizeHist(imagem_cinza)
    return imagem_equalizada

# Carrega dataset de treino da pasta "Treino"
def carregar_dataset(pasta):
    X = []
    y = []
    for nome_arquivo in os.listdir(pasta):
        if not nome_arquivo.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        imagem = preprocessar_imagem_treino(nome_arquivo)
        X.append(imagem.flatten())
        if 'cat' in nome_arquivo.lower():
            y.append('gato')
        elif 'dog' in nome_arquivo.lower():
            y.append('cachorro')
    return X, y

# Lista dos nomes das imagens de teste (na pasta "Teste")
nomes_imagens_teste = [
    'gato1.jpg', 'gato2.jpg', 'gato3.jpg',
    'cachorro1.jpg', 'cachorro2.jpg', 'cachorro3.jpg'
]

# Carregar dados de treino
X_train, y_train = carregar_dataset("Treino")

# Treinar modelo SVM
modelo = SVC(kernel='linear')
modelo.fit(X_train, y_train)

# Pré-processar e preparar dados de teste
imagens_teste_processadas = []
rotulos_teste = []

for nome in nomes_imagens_teste:
    imagem_proc = preprocessar_imagem_teste(nome)
    imagens_teste_processadas.append(imagem_proc)
    rotulo = 'gato' if 'gato' in nome else 'cachorro'
    rotulos_teste.append(rotulo)

X_test = [img.flatten() for img in imagens_teste_processadas]
y_test = rotulos_teste

# Fazer predição e avaliar
y_pred = modelo.predict(X_test)

print("=== Métricas de Avaliação ===")
print(classification_report(y_test, y_pred, digits=4))

# Mostrar imagens pré-processadas de teste com rótulos
plt.figure(figsize=(12, 6))
for i, imagem in enumerate(imagens_teste_processadas):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imagem, cmap='gray')
    plt.title(rotulos_teste[i])
    plt.axis('off')

plt.suptitle('Imagens de Teste Pré-processadas: 128x128 + Gaussiano + Equalização', fontsize=14)
plt.tight_layout()
plt.show()
