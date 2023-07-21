# all codes executed in pyhton console.

import pandas as pd

# reading the csv file

df = pd.read_csv("persona.csv")

# getting a summary of the dataframe


def check_df(dataframe, head=5):
    print("## shape ##")
    print(dataframe.shape)
    print("## types ##")
    print(dataframe.dtypes)
    print("## head ##")
    print(dataframe.head(head))
    print("## tail ##")
    print(dataframe.tail(head))
    print("## n/a values ##")
    print(dataframe.isnull().sum())
    print("## quantiles ##")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# seeing how many unique exists in SOURCE and also getting their frequency
df['SOURCE'].value_counts()
# android    2974
# ios        2026

# seeing how many unique exists in PRICE
df['PRICE'].nunique()
# 6

# seeing how many sales were made in each price
df['PRICE'].value_counts()
# 29    1305
# 39    1260
# 49    1031
# 19     992
# 59     212
# 9      200

# seeing how many sales were made in each country
df['COUNTRY'].value_counts()
# usa    2065
# bra    1496
# deu     455
# tur     451
# fra     303
# can     230

# seeing the earnings in each country
df.groupby('COUNTRY')['PRICE'].sum()
# bra    51354
# can     7730
# deu    15485
# fra    10177
# tur    15689
# usa    70225

# seeing how many sales were made in each source
df['SOURCE'].value_counts()
# android    2974
# ios        2026

# seeing price means for each country
df.groupby('COUNTRY')['PRICE'].mean()
# bra    34.327540
# can    33.608696
# deu    34.032967
# fra    33.587459
# tur    34.787140
# usa    34.007264

# seeing price means for each source
df.groupby('SOURCE')['PRICE'].mean()
# android    34.174849
# ios        34.069102

# seeing price means for both country-source values
df.groupby(['COUNTRY', 'SOURCE']).agg({'PRICE': 'mean'})
# bra     android  34.387029
#         ios      34.222222
# can     android  33.330709
#         ios      33.951456
# deu     android  33.869888
#         ios      34.268817
# fra     android  34.312500
#         ios      32.776224
# tur     android  36.229437
#         ios      33.272727
# usa     android  33.760357
#         ios      34.371703

# seeing price means for each country-source-sex-age values
df.groupby(['COUNTRY', 'SOURCE', 'SEX', 'AGE']).agg({'PRICE': 'mean'})

# sorting the output according to price variable in descending order and saving it as a new dataframe
agg_df = df.groupby(['COUNTRY', 'SOURCE', 'SEX', 'AGE']).agg({'PRICE': 'mean'}).sort_values(by='PRICE', ascending=False)

# resetting the indexes 
agg_df.reset_index(inplace=True)

# categorising the ages based on age intervals
agg_df["AGE_CAT"] = pd.cut(x=agg_df['AGE'], bins=[0, 18, 23, 30, 40, 70],
                           labels=['0_18', '19_23', '24_30', '31_40', '41_70'])

# creating a persona for each customer type
agg_df['customer_level_based'] = [f"{agg_df.iloc[i,0]}_{agg_df.iloc[i,1]}_{agg_df.iloc[i,2]}_{agg_df.iloc[i,5]}".upper()
                                  for i in range(len(agg_df))]

# getting mean of price based on customer_level_based variable
agg_df['PRICE'] = agg_df.groupby('customer_level_based')['PRICE'].transform('mean')

# creating segment values based on prices
agg_df['SEGMENT'] = pd.qcut(agg_df['PRICE'], 4, labels=['D', 'C', 'B', 'A'])

# getting rid of duplicate data (and also old indexes as well)
agg_df.drop_duplicates('customer_level_based', keep='first', inplace=True, ignore_index=True)

# seeing some info about segments
agg_df.groupby('SEGMENT').agg({'PRICE': ['mean', 'max', 'sum']})

# executing some examples to have an idea of what
# could be the income

tr_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == tr_user]

fra_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == fra_user]
