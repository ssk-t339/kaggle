exp_var = 'MedInc'
tar_var = 'HousingPrices'

# 散布図を表示
plt.figure(figsize=(12, 9))
plt.scatter(data[exp_var], data[tar_var])
plt.xlabel(exp_var)
plt.ylabel(tar_var)
plt.title('california_housing_scatter')
plt.savefig('california_housing_scatter.png')
data[['MedInc', 'HousingPrices']].describe()

# 外れ値を除去
q_95 = data['MedInc'].quantile(0.95)
print('95%点の分位数', q_95)

# 絞り込む
data = data[data['MedInc'] < q_95]

# 散布図を表示
plt.figure(figsize=(12, 9))
plt.scatter(data[exp_var], data[tar_var])
plt.xlabel(exp_var)
plt.ylabel(tar_var)
plt.title('california_housing_scatter2')
plt.savefig('california_housing_scatter2.png')
plt.show()

# 記述統計量を確認
data[['MedInc', 'HousingPrices']].describe()

# 絞り込む
data = data[data['MedInc'] < q_95]

# 散布図を表示
plt.figure(figsize=(12, 9))
plt.scatter(data[exp_var], data[tar_var])
plt.xlabel(exp_var)
plt.ylabel(tar_var)
plt.savefig('california_housing_scatter2.png')
data[['MedInc', 'HousingPrices']].describe()

# 説明変数と目的変数にデータを分割
X = data[[exp_var]]
print(data.shape)
display(X.head())
y = data[[tar_var]]
print(y.shape)
display(y.head())

# 学習
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# 精度の確認
print('回帰直線の切片', model.intercept_[0])
print('回帰係数', model.coef_[0][0])
print('決定係数', model.score(X, y))
print('回帰直線', 'y = ', model.coef_[0][0], 'x + ', model.intercept_[0])

# 回帰直線と散布図を表示
plt.figure(figsize=(12, 9))
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel(exp_var)
plt.ylabel(tar_var)
plt.title('california_housing_regression')
plt.savefig('california_housing_regression.png')