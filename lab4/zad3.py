import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# pregnant-times,glucose-concentr,blood-pressure,skin-thickness,insulin,mass-index,pedigree-func,age,class
# 6,148,72,35,0,33.6,0.627,50,tested_positive
# 1,85,66,29,0,26.6,0.351,31,tested_negative
# 8,183,64,0,0,23.3,0.672,32,tested_positive

df = pd.read_csv("diabetes.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7)
# dzielimy na dane i etykiety
train_data = train_set[:, 0:8]
train_labels = train_set[:, 8]
test_data = test_set[:, 0:8]
test_labels = test_set[:, 8]
# 0 dla negative i 1 dla positive
train_labels = list(map(lambda x: 1 if x == 'tested_positive' else 0, train_labels))
test_labels = list(map(lambda x: 1 if x == 'tested_positive' else 0, test_labels))

# przeskalujemy dane dla lepszej wydajnosci sieci neuronowej
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# tworzymy i ternujemy siec
mlp = MLPClassifier(hidden_layer_sizes=(6,3,), activation='relu', max_iter=500)
mlp.fit(train_data, train_labels)

#ewaluacja na zbiorze testowym
print(f"Pierwszy model sieci z podwojna warstwa 6,3 neuronow")
# testowy
predictions_test = mlp.predict(test_data)
print(f"Dokładność: {accuracy_score(predictions_test, test_labels)}")
print(confusion_matrix(predictions_test, test_labels))
# print(classification_report(predictions_test, test_labels))


# stworzymy jeszcze jedna siec
mlp2 = MLPClassifier(hidden_layer_sizes=(64,32,), activation='tanh', max_iter=500)
mlp2.fit(train_data, train_labels)
print(f"Drugi model sieci z podwojna warstwa 64, 32 neuronow z funkcja aktywacji tanh")
# testowy
predictions_test2 = mlp2.predict(test_data)
print(f"Dokładność: {accuracy_score(predictions_test2, test_labels)}")
print(confusion_matrix(predictions_test2, test_labels))
print(classification_report(predictions_test2, test_labels))


