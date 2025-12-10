import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam

# --- CONFIGURAÇÕES E PRÉ-PROCESSAMENTO ---
DATA_DIR = 'cell_images'
CATEGORIES = ['Uninfected', 'Parasitized'] # 0 = Uninfected, 1 = Parasitized
IMG_SIZE = 64 # Redimensionamento 64x64

def load_data():
    data = []
    labels = []
    
    print("Carregando imagens... Isso pode levar alguns instantes.")
    
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category) # 0 ou 1
        
        # Verificação básica se a pasta existe
        if not os.path.exists(path):
            print(f"Erro: Pasta não encontrada: {path}")
            continue
            
        count = 0
        for img_name in os.listdir(path):
            try:
                # Ler a imagem
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                
                if img_array is None:
                    continue
                
                # Conversão para escala de cinza
                # A escala de cinza reduz a entrada de 64x64x3 para 64x64x1 (4096 neurônios)
                gray_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                
                # Redimensionamento
                resized_array = cv2.resize(gray_array, (IMG_SIZE, IMG_SIZE))
                
                data.append(resized_array)
                labels.append(class_num)
                count += 1
            except Exception as e:
                pass
        print(f"Carregadas {count} imagens de {category}")

    # Normalização dos pixels para [0, 1]
    X = np.array(data) / 255.0
    y = np.array(labels)
    
    # Flattening para entrada na MLP (Vetor de pixels) 
    # Transforma (N, 64, 64) em (N, 4096)
    X = X.reshape(X.shape[0], -1)
    
    return X, y

# Carregar dados
X, y = load_data()

# Divisão Treino e Teste (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Tamanho do Treino: {X_train.shape}")
print(f"Tamanho do Teste: {X_test.shape}")

# --- MODELO MLP ---
model = Sequential()

# Camada de Entrada (implicita pelo input_shape) e Camadas Ocultas
# O PDF sugere 1 a 3 camadas ocultas com ReLU
model.add(Input(shape=(IMG_SIZE * IMG_SIZE,))) # 4096 neurônios de entrada
model.add(Dense(512, activation='relu'))       # Camada Oculta 1
model.add(Dense(256, activation='relu'))       # Camada Oculta 2
model.add(Dense(128, activation='relu'))       # Camada Oculta 3

# Camada de Saída: 1 neurônio, sigmoide
model.add(Dense(1, activation='sigmoid'))

# Compilação
# Otimizador Adam, Loss Binary Crossentropy
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- TREINAMENTO ---
# Treinamento com validação usando os dados de teste para plotar curvas
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# --- AVALIAÇÃO E ENTREGÁVEIS ---

# 1. Curva de perda (Loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Treino Loss')
plt.plot(history.history['val_loss'], label='Validação Loss')
plt.title('Curva de Perda (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

# Curva de Acurácia (Extra para análise)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Treino Acurácia')
plt.plot(history.history['val_accuracy'], label='Validação Acurácia')
plt.title('Curva de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_loss_acuracia.png')
plt.show()

# Predições para métricas
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

# 2. Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.savefig('matriz_confusao.png')
plt.show()

# Relatório de Acurácia
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# 3. Curva ROC e AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('curva_roc.png')
plt.show()