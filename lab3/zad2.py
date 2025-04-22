import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

df = pd.read_csv('iris1.csv')

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=292654)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]
# print(train_set)
# print(test_set)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_inputs, train_classes)

tree.plot_tree(clf)

plt.figure()
plot_tree(clf, filled=True)
plt.title("Drzewo decyzyjny dla irysów")
plt.show()

predictions = clf.predict(test_inputs)
accuracy = accuracy_score(test_classes, predictions)
print(f'Dokładność klasyfikatora: {accuracy:.2%}')

cm = confusion_matrix(test_classes, predictions)
print("Macierz błędów:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()