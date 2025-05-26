
# Prova Visão Computacional – Classificação de Imagens com Visão Computacional e IA

## Descrição do Problema

O Objetoivo deste projeto foi implementar um programa que realize a classificação de imagens de gatos e cachorros. Utilizando a prendizado de maquina e com os tratamentos de imagens com funções que vimos em sala.


## Justificativa das Técnicas Utilizadas

   - Pré-processamento com OpenCV: As imagens passaram por redimensionamento, suavização com filtro Gaussiano e equalização de histograma para melhorar a          uniformidade da iluminação e realçar detalhes relevantes, como vimos em sala ao longo do semestre.
   - Conversão para tons de cinza: Reduz complexidade e foca em formas e padrões, tornando mais simples de diferenciar as imagens.
   - Flattening das imagens: Como forma simples de representar cada imagem como vetor de características.
   - Classificador SVM: Escolhi por ser simples de se usar e por já termos utilizados em outras matérias na faculdade, o SVM tenta separar as classes das          imagens com uma linha, nesse caso como estamos tratando de apenas duas categorias, sendo cachorro e gato, o SVM se torna um bom exemplo de classificador.
   - Avaliação com métricas clássicas: Precisão, recall e F1-score para analisar a performance da classificação, são as métricas padrões que mudamos quando        estamos aprendendo aprendizado de máquina.

## Etapas Realizadas

1. Carregamento das imagens da pasta imagens,  sendo 3 de gatos e 3 de cachorros.

2. Pré-processamento:
   - Redimensionamento para 128x128 com cv2.resize
   - Filtro Gaussiano (5x5) com cv2.GaussianBlur
   - Conversão para escala de cinza com cv2.cvtColor(imagem_suavizada, cv2.COLOR_BGR2GRAY)
   - Equalização de histograma com cv2.equalizeHist

3. Visualização
   - Retornamos a imagem da função de processamento e usando a biblioteca matplotlib trazemos as imagens alteradas na tela

4. Transformação das imagens em vetores flatten, pois o SVM trabalha com dado em forma de vetor, não com a imagem em si.

5. Divisão dos dados em treino (80%) e teste (20%) com train_test_split, trazendo stratufy=y para que a quantidade de gatos e cachorros
   em cada lado seja semore a mesma.

6. Treinamento e Classificação com um modelo SVM
   - Carregar dataset le todas as imagens do dataset, faz o pré-processamento delas e transforma cada imagem em um vetor.
   - Esses dados são então usados para treinar o modelo SVM.
   - Então pré-processamos as imagens da pasta Teste.
   - O modelo treinado então faz a predição para cada valor do vetor de Teste, para verificar se é gato ou cachorro.
   - Compara essas previsões do modelo com os rótulos reais

7. Resultados
   - Exibe as métricas com a avaliação do modelo, sendo elas, precisão, recall e F1-score.
   - Traz as imagens pré-processadas com seus rótulos reais, 

## Resultados Obtidos

O modelo treinado conseguiu classificar corretamente as imagens de teste. Abaixo está um exemplo das métricas de avaliação:

```
=== Métricas de Avaliação ===
              precision    recall  f1-score   support

    cachorro     0.6667    0.6667    0.6667         3
        gato     0.6667    0.6667    0.6667         3

    accuracy                         0.6667         6
   macro avg     0.6667    0.6667    0.6667         6
weighted avg     0.6667    0.6667    0.6667         6
```

## Tempo Total Gasto
   
   - Demorou cerca 1 hora e 40 minutos para o desenvolvimmento e documentação. 

## Dificuldades Encontradas

   - Trazer o dataset para treinamento, pois a biblioteca passada não funcionou, então foi necessário baixar um a parte para fazer o
     treinamento de forma correta.   
