import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('iris1.csv')

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=292654)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]
# print(train_set)
# print(test_set)


k_values = [3, 5, 11]

for k in k_values:
    print(f"\n🔹 Wyniki dla k-NN (k={k})")

    knn = KNeighborsClassifier(n_neighbors=k)  # Inicjalizacja klasyfikatora k-NN
    knn.fit(train_inputs, train_classes)  # Trenowanie modelu

    predictions = knn.predict(test_inputs)  # Predykcja na zbiorze testowym
    accuracy = accuracy_score(test_classes, predictions)  # Obliczenie dokładności

    print(f'Dokładność: {accuracy:.2%}')

    # Macierz błędów
    cm = confusion_matrix(test_classes, predictions)
    print("Macierz błędów:\n", cm)

    #  Wizualizacja macierzy błędów
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Macierz błędów k-NN (k={k})')
    plt.show()


print("\n🔹 Wyniki dla Naive Bayes")

nb = GaussianNB()  # Inicjalizacja klasyfikatora Naive Bayes
nb.fit(train_inputs, train_classes)  # Trenowanie modelu

predictions_nb = nb.predict(test_inputs)  # Predykcja na zbiorze testowym
accuracy_nb = accuracy_score(test_classes, predictions_nb)  # Obliczenie dokładności

print(f'Dokładność: {accuracy_nb:.2%}')

# Macierz błędów
cm_nb = confusion_matrix(test_classes, predictions_nb)
print("Macierz błędów:\n", cm_nb)

# Wizualizacja macierzy błędów
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot()
plt.title('Macierz błędów Naive Bayes')
plt.show()