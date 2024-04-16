import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


dataset_path = 'market_basket_dataset.csv'
data = pd.read_csv(dataset_path)


print("Dataset Information:")
print(data.info())


data.dropna(inplace=True)


data_encoded = pd.get_dummies(data)


min_support = 0.05  
frequent_itemsets = apriori(data_encoded, min_support=min_support, use_colnames=True)


association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)


print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(association_rules_df)
