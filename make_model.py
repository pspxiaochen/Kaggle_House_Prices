import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# 准备特征工程
all_df = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],test_df.loc[:,'MSSubClass':'SaleCondition':]),axis=0,ignore_index=True)
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
quantitative = [i for i in all_df.columns if all_df.dtypes[i] != object]
qualitative = [i for i in all_df.columns if all_df.dtypes[i] == object]

# 处理缺失数据
missing = all_df.isnull().sum()
missing.sort_values(inplace=True,ascending=False)
all_df = all_df.drop(missing.loc[missing>1].index,axis = 1)

#处理log项
logFeatures = ['GrLivArea','1stFlrSF','2ndFlrSF','TotalBsmtSF','LotArea','KitchenAbvGr','GarageArea']
for logFeature in logFeatures:
    all_df[logFeature]=np.log1p(all_df[logFeature])

#添加一些boolean变量
all_df['HasBasement'] = all_df['TotalBsmtSF'].apply(lambda x:1 if x > 0 else 0)
all_df['HasGarage'] = all_df['GarageArea'].apply(lambda x:1 if x > 0 else 0)
all_df['Has2ndFloor'] = all_df['2ndFlrSF'].apply(lambda x:1 if x > 0 else 0)
all_df['Has2ndKitchen'] = all_df['KitchenAbvGr'].apply(lambda x:1 if x>1 else 0)
all_df['HasWoodDeck'] = all_df['WoodDeckSF'].apply(lambda x:1 if x>0 else 0 )
all_df['HasPorch'] = all_df['OpenPorchSF'].apply(lambda x:1 if x > 0 else 0)
all_df['HasPool'] = all_df['PoolArea'].apply(lambda x:1 if x > 0 else 0)
all_df['IsNew'] = all_df['YearBuilt'].apply(lambda x:1 if x >= 2000 else 0)

quantitative = [f for f in all_df.columns if all_df.dtypes[f] != 'object']
qualitative = [f for f in all_df.columns if all_df.dtypes[f] == 'object']


# 对于定性变量的encode
all_dummy_df = pd.get_dummies(all_df)

# 对于数值变量进行标准化
mean_cols = all_dummy_df.mean()
all_dummy_df=all_dummy_df.fillna(mean_cols)

X = all_dummy_df[quantitative]
std = StandardScaler()
s = std.fit_transform(X)

all_dummy_df[quantitative] = s

dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

y_train = np.log(train_df['SalePrice'])

#模型预测
#岭回归
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model,dummy_train_df,y_train.values,scoring='neg_mean_squared_error',cv=5))
    return rmse

# alphas = np.logspace(start=-3,stop=2,num=50,base=10)  # 在-3到2之间 取50个 都在10的-3次方和10的2次方之间
# cv_ridge = []
# coefs = []
# for alpha in alphas:
#     model = Ridge(alpha=alpha)
#     model.fit(dummy_train_df,y_train)
#     cv_ridge.append(rmse_cv(model).mean())
#     coefs.append(model.coef_)

# cv_ridge = pd.Series(cv_ridge,index=alphas)
# cv_ridge.plot(title = "Validation - Just Do It")
# plt.xlabel('alpha')
# plt.ylabel('rmse')
# #plt.show()
#print(cv_ridge)

#稀疏约束 Lasso
from sklearn.linear_model import Lasso,LassoCV
# alphas = np.logspace(-4,-2,100)
# cv_lasso = []
# coefs = []
# for alpha in alphas:
#     model = Lasso(alpha = alpha,max_iter=5000)
#     model.fit(dummy_train_df,y_train)
#     cv_lasso.append(rmse_cv(model).mean())
#     coefs.append(model.coef_)

# cv_lasso = pd.Series(cv_lasso,index=alphas)
# cv_lasso.plot(title = "Validation - Just Do It")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()
#print(cv_lasso.min(), cv_lasso.argmin()) #argmin 返回最小值的索引
model = Lasso(alpha = 0.00058,max_iter=5000)
model.fit(dummy_train_df,y_train)

# Lasso(alpha=0.00058, copy_X=True, fit_intercept=True, max_iter=5000,
#    normalize=False, positive=False, precompute=False, random_state=None,
#    selection='cyclic', tol=0.0001, warm_start=False)

coef = pd.Series(model.coef_, index = dummy_train_df.columns)
# print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# imp_coef = pd.concat([coef.sort_values().head(10),
#                      coef.sort_values().tail(10)])
# matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
# imp_coef.plot(kind = "barh")
# plt.title("Coefficients in the Lasso Model")
# plt.show()

# 特征二
import utils
train_df_munged,label_df,test_df_munged = utils.feature_engineering()
print(train_df_munged)






