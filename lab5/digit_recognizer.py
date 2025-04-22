import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.h5',        # nazwa pliku
    monitor='val_accuracy', # co monitorujemy
    save_best_only=True,    # zapis tylko gdy się poprawi
    mode='max',             # szukamy maksimum accuracy
    verbose=1               # wypisuje info przy zapisie
)

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)  # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2,
          callbacks=[history, checkpoint])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()


# a) Co się dzieje w preprocessing? Do czego służy funkcja reshape, to_categorical i np.argmax?

# train_images.reshape((train_images.shape[0], 28, 28, 1)) – dodajemy kanał (1) na końcu, by format danych pasował do sieci CNN (która oczekuje obrazów z kanałem, np. 1 dla czarno-białych, 3 dla RGB).
# Wymiary zmieniają się z (60000, 28, 28) → (60000, 28, 28, 1)

# .astype('float32') / 255 – skalujemy wartości pikseli z zakresu [0, 255] do [0, 1], co przyspiesza i stabilizuje trening.

# to_categorical – zamienia etykiety (np. 3) na wektory one-hot:
# 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# np.argmax(test_labels, axis=1) – odwrotność one-hot: bierze indeks, gdzie jest 1.
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] → 3

# b) Jak dane przepływają przez sieć i jak się w niej transformują?

# Warstwa 1 – Conv2D(32, (3,3)):
# Przetwarza obraz 28x28x1 filtrami 3x3 → tworzy 32 mapy cech (feature maps), wynik ma rozmiar 26x26x32.

# Warstwa 2 – MaxPooling2D((2,2)):
# Zmniejsza każdy wymiar przestrzenny o połowę: 26x26 → 13x13 (na każdej z 32 map cech).

# Warstwa 3 – Flatten():
# Spłaszcza dane 13x13x32 → 5408-elementowy wektor, gotowy do Dense.

# Warstwa 4 – Dense(64):
# Klasyczna warstwa w pełni połączona – dostaje 5408 wejść, zwraca 64 aktywacje (neurony).

# Warstwa 5 – Dense(10):
# Warstwa wyjściowa, po 1 neuronie na każdą cyfrę (0–9), softmax przelicza na prawdopodobieństwa.

# c) Jakich błędów na macierzy pomyłek jest najwięcej i dlaczego?
# 7 i 9
# 9 i 4
# d) Co możesz powiedzieć o krzywych uczenia się. Czy mamy
# przypadek przeuczenia lub niedouczenia się?
# e) Jak zmodyfikować kod programu, aby model sieci był
# zapisywany do pliku h5 co epokę, pod warunkiem, że w tej
# epoce osiągnęliśmy lepszy wynik?