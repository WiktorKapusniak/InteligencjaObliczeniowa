import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3,
                                                    random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('iris_model.keras')

# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# a) Co robi StandardScaler? Jak transformowane są dane liczbowe?
# StandardScaler przekształca dane tak, by każda cecha (kolumna) miała średnią 0 i odchylenie standardowe 1.
# Przykład: jeśli mamy cechę "długość płatka", to po skalowaniu wartości są w nowej skali, gdzie:
# średnia = 0, a rozrzut danych dopasowany tak, by odchylenie standardowe = 1.
# To bardzo pomaga w uczeniu, bo sieci lepiej uczą się, gdy dane wejściowe mają podobny rozkład.

# b) Czym jest OneHotEncoder (i kodowanie „one hot” ogólnie)? Jak etykiety klas są transformowane przez ten encoder?
# OneHotEncoder zamienia etykiety klas (np. 0, 1, 2) na wektory binarne.
# Przykład: jeśli mamy 3 klasy (Iris-setosa, Iris-versicolor, Iris-virginica), to:
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]
# Taka reprezentacja zapobiega traktowaniu klas jako liczb porządkowych (czyli np. że 2 > 0).

# c) Ile neuronów ma warstwa wejściowa i wyjściowa?
# input_shape=(X_train.shape[1],) oznacza, że sieć oczekuje wektora o długości 4 (bo Iris ma 4 cechy).
# Czyli warstwa wejściowa ma 4 wejścia (cechy: długość i szerokość działki kielicha i płatka).
# Warstwa wyjściowa ma y_encoded.shape[1] = 3 neurony (bo mamy 3 klasy do rozróżnienia).
# Użycie softmax w wyjściu daje nam rozkład prawdopodobieństwa – model mówi, jak bardzo "wierzy", że dana próbka należy do każdej klasy.

# d) Czy funkcja aktywacji relu jest najlepsza do tego zadania?
# ReLU (Rectified Linear Unit) to domyślny wybór, bo jest szybki w obliczeniach i dobrze działa w wielu przypadkach.
# W wielu przypadkach 'tanh' działa lepiej przy małych zestawach danych, bo wartości są między -1 a 1.

# e) Czy różne optymalizatory lub funkcje straty dają różne wyniki?

# Różne funkcje straty pasują do różnych typów problemów:
# - categorical_crossentropy → klasy wielokrotne, one-hot
# - sparse_categorical_crossentropy → klasy jako liczby (bez one-hot)
# - mean_squared_error → regresja
# W tym przypadku categorical_crossentropy jest poprawna, bo etykiety są one-hot.
