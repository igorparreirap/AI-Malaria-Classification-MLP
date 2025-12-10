# AI Malaria Classification MLP

Este projeto consiste no desenvolvimento de um pipeline completo de classificação de imagens microscópicas para distinguir células sanguíneas infectadas por malária (*Parasitized*) de células não infectadas (*Uninfected*). O modelo utilizado é uma Rede Neural Artificial do tipo Multilayer Perceptron (MLP).

Trabalho desenvolvido como requisito da disciplina de Inteligência Artificial (2025.2) do curso de Engenharia de Computação da PUC Goiás.

## 📋 Estrutura do Projeto

- `data/`: Instruções sobre o dataset utilizado.
- `reports/`: Relatório técnico final em PDF com a análise detalhada dos resultados.
- `src/`: Código fonte em Python contendo o pré-processamento, modelo MLP e geração de gráficos.

## 🚀 Tecnologias Utilizadas

- **Linguagem:** Python 3.x
- **Bibliotecas:** TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Seaborn, Scikit-learn.

## 📊 Resultados Obtidos

O modelo MLP atingiu uma acurácia global de **85%**.
- **Arquitetura:** Entrada (4096) -> Dense(512) -> Dense(256) -> Dense(128) -> Saída(1).
- **Métricas:** O relatório completo com Matriz de Confusão e Curva ROC encontra-se na pasta `reports/`.

## ⚙️ Como Executar

1. Clone o repositório:
   ```bash
   git clone [https://github.com/igorparreirap/AI-Malaria-Classification-MLP.git](https://github.com/igorparreirap/AI-Malaria-Classification-MLP.git)
