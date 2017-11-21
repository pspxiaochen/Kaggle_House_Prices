import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("train.csv")
print(df_train.MSSubClass)
df_train['SalePrice'].describe()
# #直方图
# sns.distplot(df_train['SalePrice'])
# #峰度和偏度
# # print("Skewness: %f" % df_train['SalePrice'].skew())
# # print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#
# ####GrLivArea: 地面以上生活区的尺寸
# # 散点图：grlivarea/saleprice
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
# data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
# ####TotalBsmtSF:地下室的总面积
# var = 'TotalBsmtSF'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
# data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#
# ####OverallQual:评估房子的整体材料
# var = 'OverallQual'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
# f,ax = plt.subplots(figsize = (8,6))
# fig = sns.boxplot(x=var,y='SalePrice',data=data)
# fig.axis(ymin=0,ymax=800000)
# ####YearBuilt: 房子开始施工的日期
# var = 'YearBuilt'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
# f.ax = plt.subplots(figsize=(16,8))
# fig = sns.boxplot(x=var,y='SalePrice',data=data)
# fig.axis(ymin=0,ymax=800000)
# #将横坐标的年份旋转90度
# plt.xticks(rotation = 90)
#
# #LotArea: 房屋占土地平方英尺
# var = 'LotArea'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
# data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000),xlim=(0,20000))
#
# ####相关矩阵
# corrmat = df_train.corr() #查看相关性
# f , ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat,cmap='RdBu',vmax=.8,square=True)
# plt.xticks(rotation = 90)
# plt.yticks(rotation = 0)
#
# ####“SalePrice”相关矩阵
# k = 10 #热力图的变量数
# cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)  # 计算相关系数
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
# plt.yticks(rotation = 0)
#
# #散点图
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols],size=2.5)
# #plt.show()

#missing data
total =df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
df_train = df_train.drop((missing_data[missing_data['Total']>1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

####标准化数据
salePrice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
np.set_printoptions(threshold=np.NaN)
#print(df_train['SalePrice'][:,np.newaxis])
low_range = salePrice_scaled[salePrice_scaled[:,0].argsort()][:10] # .argsort(): 得到从小到大的索引
high_range = salePrice_scaled[salePrice_scaled[:,0].argsort()][-10:]
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution:')
# print(high_range)

####双变量分析
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
# data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))

#有2个点很特殊 我们需要删除他  有2个点的GrLivArea很大，但是price却很低
#print(df_train.sort_values(by='GrLivArea',ascending=False)[:2])
df_train=df_train.drop(df_train[df_train['Id'] == 1298].index)
df_train=df_train.drop(df_train[df_train['Id'] == 523].index)

#直方图和概率分布图
# sns.distplot(df_train['SalePrice'],fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()

#应用指数变换
df_train['SalePrice'] = np.log(df_train['SalePrice'])

# sns.distplot(df_train['SalePrice'],fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'],plot=plt)
# plt.show()

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

df_train = pd.get_dummies(df_train)
print(df_train.shape)
#print(df_train.OverallCond)