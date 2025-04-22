import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

file_path = "iris1.csv"
df = pd.read_csv(file_path)

sepal_length = df['sepal.length']
sepal_width = df['sepal.width']
target = df['variety']

colors = {'Setosa': 'r', 'Versicolor': 'g', 'Virginica': 'b'}
#oryginalne
plt.figure(figsize=(8, 6))
for variety in colors:
    variety_data = df[df['variety'] == variety]
    plt.scatter(variety_data['sepal.length'], variety_data['sepal.width'], 
                color=colors[variety], label=variety)

plt.title('Sepal Length vs Sepal Width (Original)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.savefig("iris_sepal_original.png")
plt.show()

# Normalizacja Min-Max
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[['sepal.length', 'sepal.width']] = scaler.fit_transform(df[['sepal.length', 'sepal.width']])

plt.figure(figsize=(8, 6))
for variety in colors:
    variety_data = df_normalized[df_normalized['variety'] == variety]
    plt.scatter(variety_data['sepal.length'], variety_data['sepal.width'], 
                color=colors[variety], label=variety)

plt.title('Sepal Length vs Sepal Width (Normalized - Min-Max)')
plt.xlabel('Sepal Length (Normalized)')
plt.ylabel('Sepal Width (Normalized)')
plt.legend()
plt.savefig("iris_sepal_normalized.png")
plt.show()

# Skalowanie Z-score
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[['sepal.length', 'sepal.width']] = scaler.fit_transform(df[['sepal.length', 'sepal.width']])

plt.figure(figsize=(8, 6))
for variety in colors:
    variety_data = df_standardized[df_standardized['variety'] == variety]
    plt.scatter(variety_data['sepal.length'], variety_data['sepal.width'], 
                color=colors[variety], label=variety)

plt.title('Sepal Length vs Sepal Width (Standardized - Z-score)')
plt.xlabel('Sepal Length (Standardized)')
plt.ylabel('Sepal Width (Standardized)')
plt.legend()
plt.savefig("iris_sepal_standardized.png")
plt.show()

print("Original Data Stats:")
print(df[['sepal.length', 'sepal.width']].describe())

print("\nNormalized Data Stats:")
print(df_normalized[['sepal.length', 'sepal.width']].describe())

print("\nStandardized Data Stats:")
print(df_standardized[['sepal.length', 'sepal.width']].describe())
