import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv('diabetes.csv')


# exploratory data analysis


def check_df(dataframe, head=5):
    """
    a function to read the dataframe's basic info and print.
    Parameters
    ----------
    dataframe: dataframe
       dataframe that contains the variables.
    head: int
       parameter to show the desired amount of value to user. initally set to 5.

    Returns
    -------

    """
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


def grab_col_names(dataframe, car_th=20, cat_th=10):
    """
   function to find cardinal and categorical variables in a dataframe

  Parameters
  ----------
  dataframe: dataframe
    dataframe that contains the variables
  car_th: int, float
    threshold for numerical but actually categorical values
  cat_th: int, float
     threshold for categorical but actually numerical values

  Returns
  -------
  cat_cols: list
    categorical variables list
  num_cols: list
    numerical variables list
  cat_but_car: list
    categorical but actually numerical variables list

    notes
    ------
    cat_cols + num_cols + cat_but_car = sum of all the variables
    num_but_car is already in the cat_cols
  """

    col_names = ["object", "category", "bool"]

    cat_cols = [col for col in dataframe.columns if
                (str(dataframe[col].dtypes) in col_names and dataframe[col].nunique() < cat_th)]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].nunique() < cat_th) and
                   str(dataframe[col].dtypes) not in col_names]

    num_cols = [col for col in dataframe.columns if
                dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) not in col_names]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in
                   ["object", "category"]]

    cat_cols = cat_cols + num_but_cat

    print(f"observations: {dataframe.shape[0]}")
    print(f"variables: {dataframe.shape[1]}")
    print(f"cat cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

    print("#####")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "ratio:": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quar_1 = dataframe[col_name].quantile(q1)
    quar_3 = dataframe[col_name].quantile(q3)
    iqr = quar_3 - quar_1
    upper = quar_3 + 1.5 * iqr
    lower = quar_1 - 1.5 * iqr
    return lower, upper


def check_outliers(dataframe, col_name, low, up):
    return dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None)



def replace_outliers(dataframe, variable, low, up):
    dataframe.loc[(dataframe[variable] < low, variable)] = low
    dataframe.loc[(dataframe[variable] > up, variable)] = up



# getting basic info of the dataframe
check_df(df)
# grouping the categorical and cardinal columns.
cat_cols, num_cols, cat_but_car = grab_col_names(df, car_th=10)

# getting summaries based on types of the values.
for col in cat_cols:
    cat_summary(df, col)
for col in num_cols:
    num_summary(df, col)

# getting target summary based on numerical variables.
# the target variable is 'outcome', which is actually the only element present in the
# cat_cols, so there's no need to get a target summary with categorical variables.
for col in num_cols:
    target_summary_with_num(df, 'Outcome', col)

# checking to see if there are outliers present in each variable.
for col in num_cols:
    lower, upper = outlier_thresholds(df, col)
    print(col, check_outliers(df,col, lower, upper))

# checking to see if there are null values.
df.isnull().sum()

# doing a correlation analysis.
df.corr()

# since there are outliers present in every variable, replacing them
# with outlier thresholds.
for col in num_cols:
    lower, upper = outlier_thresholds(df, col)
    if check_outliers(df, col, lower, upper):
        replace_outliers(df, col, lower, upper)


# while there's no missing values present in the dataframe, there are some variables
# which cannot have 0 as a value, such as blood pressure or insulin. getting a list of
# these variables manually and change the zeros with the mean of their own variable.
zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def replace_zeros(dataframe, column):
    fill_value = dataframe[column].mean()
    dataframe.loc[(dataframe[column] == 0, column)] = fill_value


for col in zero_values:
    replace_zeros(df, col)

# checking if the process was successful
for col in zero_values:
    num_summary(df, col)

# doing some feature extraction
df.loc[((df['Age'] <= 44) & (df['Pregnancies'] == 0)), 'NEW_AGE_PREGNANCIES'] = 'YOUNG_NO_CHILD'
df.loc[((df['Age'] <= 44) & (df['Pregnancies'] > 0) & (df['Pregnancies'] <= 3)), 'NEW_AGE_PREGNANCIES'] = 'YOUNG_PREG_1_3'
df.loc[((df['Age'] <= 44) & (df['Pregnancies'] > 3)), 'NEW_AGE_PREGNANCIES'] = 'YOUNG_PREG_4'
df.loc[((df['Age'] > 44) & (df['Pregnancies'] == 0)), 'NEW_AGE_PREGNANCIES'] = 'OLD_NO_CHILD'
df.loc[((df['Age'] > 44) & (df['Pregnancies'] > 0) & (df['Pregnancies'] <= 3)), 'NEW_AGE_PREGNANCIES'] = 'OLD_PREG_1_3'
df.loc[((df['Age'] > 44) & (df['Pregnancies'] > 3)), 'NEW_AGE_PREGNANCIES'] = 'OLD_PREG_4'

df.loc[((df['Age'] <= 44) & (df['BMI'] < 18.5)), 'NEW_AGE_BMI'] = 'YOUNG_LOW_BMI'
df.loc[((df['Age'] <= 44) & (df['BMI'] >= 18.5) & (df['BMI'] <= 24.9)), 'NEW_AGE_BMI'] = 'YOUNG_NORMAL_BMI'
df.loc[((df['Age'] <= 44) & (df['BMI'] > 24.9)), 'NEW_AGE_BMI'] = 'YOUNG_HIGH_BMI'
df.loc[((df['Age'] > 44) & (df['BMI'] < 18.5)), 'NEW_AGE_BMI'] = 'OLD_LOW_BMI'
df.loc[((df['Age'] > 44) & (df['BMI'] >= 18.5) & (df['BMI'] <= 24.9)), 'NEW_AGE_BMI'] = 'OLD_NORMAL_BMI'
df.loc[((df['Age'] > 44) & (df['BMI'] > 24.9)), 'NEW_AGE_BMI'] = 'OLD_HIGH_BMI'

df.loc[(df['Glucose'] <= 70), 'NEW_GLUCOSE'] = 'LOW_GLUCOSE'
df.loc[((df['Glucose'] > 70) & (df['Glucose'] <= 108)), 'NEW_GLUCOSE'] = 'NORMAL_GLUCOSE'
df.loc[(df['Glucose'] > 108), 'NEW_GLUCOSE'] = 'HIGH_GLUCOSE'

df.loc[(df['Age'] <= 44), 'NEW_AGE'] = 'CHILDBEARER'
df.loc[(df['Age'] > 44), 'NEW_AGE'] = 'NOT_CHILDBEARER'

# encoding the binary columns with label encoder
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
binary_cols.remove('Outcome')
df = label_encoder(df, binary_cols)

# one-hot-encoding categorical columns which are non-ordinal: 
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [column for column in df.columns if 10 >= df[column].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
df.drop(useless_cols, axis=1, inplace=True)

# standard scaling numerical cols
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()


# grouping data together
y = df['Outcome']
X = df.drop('Outcome', axis=1)

# creating the variables and the model:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# accuracy around %78.

# seeing which variables makes the most impact for the model:
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)


