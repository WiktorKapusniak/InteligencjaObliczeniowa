from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# a) dane testowe i podzial na train i test 70/30
iris = load_iris()
datasets = train_test_split(iris.data, iris.target, test_size=0.3)
train_data , test_data, train_labels, test_labels = datasets
# print(train_data[:4])
# b) nazwy przekonwertowane na liczby
# print(train_labels)
# 0 - setosa, 1 - versicolor, 2 - virginica

# c) skalowanie danych poprawia dzialanie sieci neuronowej
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
# print(train_data[:4])

# d) tworzenie i trenowanie modelu sieci neuronowej
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', max_iter=3000)
mlp.fit(train_data, train_labels)
# e) ewaluacji pierwszego modelu sieci neuronowej na zbiorze testowym
print(f"Pierwszy model sieci z warstwa 2 neuronow")
# treningowy
predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
# testowy
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))


# # f) tworzenie drugiej sieci neuronowej z 3 neuronami
mlp2 = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', max_iter=3000)
mlp2.fit(train_data, train_labels)
print(f"\nDrugi model sieci z warstwa 3 neuronow")
# treningowy
predictions_train2 = mlp2.predict(train_data)
print(accuracy_score(predictions_train2, train_labels))
# testowy
predictions_test2 = mlp2.predict(test_data)
print(accuracy_score(predictions_test2, test_labels))


# g) tworzenie trzeciej sieci neuronowej z podwojna warstwa neuronow po 3 neurony kazda
mlp3 = MLPClassifier(hidden_layer_sizes=(3,3), activation='relu', max_iter=3000)
mlp3.fit(train_data, train_labels)
# treningowy
print(f"\nTrzeci model sieci z podwojna warstwa 3 neuronow")
predictions_train3 = mlp3.predict(train_data)
print(accuracy_score(predictions_train3, train_labels))
# testowy
predictions_test3 = mlp3.predict(test_data)
print(accuracy_score(predictions_test3, test_labels))



# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(predictions_train, train_labels))
#
# print(confusion_matrix(predictions_test, test_labels))
#
# from sklearn.metrics import classification_report
# print(classification_report(predictions_test, test_labels))