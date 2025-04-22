import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('iris1.csv')

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=292654)

def classify_iris(sl, sw, pl, pw): 
    if pw < 0.8:
        return("Setosa") 
    elif pl <= 5: 
        return("Versicolor")
    else: 
        return("Virginica")
    
good_predictions = 0 
len = test_set.shape[0] 
 
for i in range(len): 
    if classify_iris(test_set[i][0],test_set[i][1],test_set[i][2],test_set[i][3]) == test_set[i][4]: 
        good_predictions = good_predictions + 1 
 
print(good_predictions) 
print(good_predictions/len*100, "%")

# Konwersja train_set na DataFrame
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
train_set_df = pd.DataFrame(train_set, columns=columns)

# Sortowanie po kolumnie 'species' (czwarta kolumna)
train_set_df = train_set_df.sort_values(by='species')

pd.set_option('display.max_rows', 110)
print(train_set_df)