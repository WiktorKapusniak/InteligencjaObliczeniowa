import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Wykonaj PCA (kompresja danych do 2 lub 3 wymiarów)
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print("Cumulative explained variance:", cumulative_variance)

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Liczba komponentów wymaganych do wyjaśnienia 95% wariancji: {n_components_95}")

if n_components_95 == 3:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, s=40)

    ax.set_title('First three PCA dimensions')
    ax.set_xlabel('1st Principal Component')
    ax.set_ylabel('2nd Principal Component')
    ax.set_zlabel('3rd Principal Component')

    plt.savefig("3D_PCA.png")

if n_components_95 == 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=40)
    plt.title('First two PCA dimensions')
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.savefig("2D_PCA.png")
