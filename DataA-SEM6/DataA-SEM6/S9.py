import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


transactions = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread', 'butter', 'jam'],
    ['eggs', 'butter', 'jam'],
    ['milk', 'bread', 'eggs', 'butter'],
    ['bread', 'butter']
]


df = pd.DataFrame(transactions)


print("Dataset Information:")
print(df.info())


df_encoded = pd.get_dummies(df.apply(lambda x: pd.Series(x), axis=1).stack()).sum(level=0)


min_support = 0.4
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)


association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)


print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(association_rules_df)
