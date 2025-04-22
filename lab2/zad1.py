import pandas as pd
from difflib import get_close_matches

file_path = "iris_with_errors.csv"
df = pd.read_csv(file_path)

#a)
for col in df.columns:
    if col != "variety":
        df[col] = pd.to_numeric(df[col], errors='coerce')
    

# print(df.describe())
# print(df.isnull().sum())

#b)
for col in df.columns:
    if col != "variety":
        mask = (df[col].isna()) | (df[col] <= 0) | (df[col] > 15)
        median_value = df[col].median(skipna=True)
        df.loc[mask, col] = median_value
# print(df.isnull().sum())

#c)
valid_varieties = ["Setosa", "Versicolor", "Virginica"]

# Funkcja do poprawy nazw na podstawie podobieństwa
def correct_variety_name(variety):
    match = get_close_matches(variety, valid_varieties, n=1, cutoff=0.6)
    return match[0] if match else variety  # Jeśli znaleziono dopasowanie, zwróć poprawioną wartość

df['variety'] = df['variety'].astype(str).apply(correct_variety_name)

df.to_csv("iris_cleaned.csv", index=False)

