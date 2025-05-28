import pandas as pd
import numpy as np
from apyori import apriori
import matplotlib.pyplot as plt
df = pd.read_csv('titanic.csv', index_col=0)
# print(df)

transactions = []

for _, row in df.iterrows():
    transaction = [
        row["Class"],
        row["Sex"],
        row["Age"],
        row["Survived"]
    ]   
    transactions.append(transaction)

# print(transactions[0])

rules = apriori(transactions, min_support=0.005, min_confidence=0.8)
results = list(rules)
# print(results[0])
# print(f'Znaleziono {len(results)} reguł.')
def inspect(results):
    rule_list = []
    for result in results:
        for stat in result.ordered_statistics:
            rule_list.append({
                'Base': tuple(stat.items_base),
                'Add': tuple(stat.items_add),
                'Support': result.support,
                'Confidence': stat.confidence,
                'Lift': stat.lift
            })
    return pd.DataFrame(rule_list)
df_results = inspect(results)
rules_df = df_results.sort_values(by='Confidence', ascending=False)


survived_yes = rules_df[rules_df['Add'].apply(lambda x: 'Yes' in x)]
survived_no = rules_df[rules_df['Add'].apply(lambda x: 'No' in x)]
plt.figure(figsize=(10, 6))
plt.scatter(rules_df['Confidence'], rules_df['Lift'], alpha=0.6)
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Lift vs Confidence - Reguły Apriori (Titanic)')
plt.grid(True)
plt.show()