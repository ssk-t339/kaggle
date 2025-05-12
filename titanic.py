import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# 欠損値の補完, ラベルエンコーディング, 列の削除
def process_df(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    return df

# データセット読み込み
df_train = pd.read_csv('../input/titanic/train.csv')
df_train = process_df(df_train)

df_test = pd.read_csv('../input/titanic/test.csv')
df_test = process_df(df_test)

# 特徴量と目的変数に分割
X_train = df_train.drop(['Survived'], axis=1)
y_train = df_train['Survived']

# データを学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# モデルの学習と評価
model = RandomForestClassifier()
model.fit(X_train , y_train)
print(f"score: {model.score(X_test, y_test)}")

# 結果の出力
y_pred = model.predict(df_test)
output = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": y_pred})
output.to_csv('submission.csv', index=False)

