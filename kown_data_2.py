import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
np.set_printoptions(threshold=np.NaN)
#把训练集和测试集除了第一个和最后一个属性不要，其他所有属性拼接到一起
all_df = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],test_df.loc[:,'MSSubClass':'SaleCondition']),axis=0,ignore_index=True)

# 将MSSubClass 属性从int 变为 str
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

# 查看所有的数值类型
quantitative = [i for i in all_df.columns if all_df.dtypes[i] != 'object']
# 查看所有的类别类型
qualitative = [i for i in all_df.columns if all_df.dtypes[i] == 'object']

#print('数值类型的个数是',len(quantitative),'类别类型的个数是',len(qualitative))

missing = all_df.isnull().sum()
missing.sort_values(inplace=True,ascending=False)
missing = missing[(missing>0)]
types = all_df[missing.index].dtypes
#计算缺失比率
percent = (all_df[missing.index].isnull().sum()/len(all_df[missing.index])*100)

missing_data = pd.concat([missing,percent,types],axis=1,keys=['Total','Percent','Type'])
#missing_data.plot.bar()

#查看相关性情况
corrMat = train_df.corr()
k = 10
cols = corrMat.nlargest(k,'SalePrice')['SalePrice'].index
#print(missing_data.index.intersection(cols))
#print(missing_data.loc[missing_data.index.intersection(cols)])
#print(missing_data.loc[(missing_data['Total']>1)].index)
#print(all_df['MSZoning'])
all_df = all_df.drop((missing_data.loc[(missing_data['Total']>1)].index),axis = 1)
print(all_df)
train_df['SalePrice'] = np.log(train_df['SalePrice'])

quantitative = [f for f in all_df.columns if all_df.dtypes[f] != 'object']
qualitative = [f for f in all_df.columns if all_df.dtypes[f] == 'object']

# 定量特征分析
train = all_df.loc[train_df.index]
train['SalePrice'] = train_df['SalePrice']

def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = qualitative #类别类型
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():  #去除重复元素
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1] #此处 stats.f_oneway 的作用是计算这种定性变量对于SalePrice的作用，如果GarageType的每个类别SalePrice的价格方差差不多，意味着该变量对于SalePrice就没什么作用
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

# a = anova(train)
# a['disparity'] = np.log(1./a['pval'].values)
# sns.barplot(data=a, x='feature', y='disparity')
# x=plt.xticks(rotation=90)
#plt.show()

#对这些定性变量进行下处理，对齐进行数值编码，让他转换为定性的列
def encode(frame,feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature,'SalePrice']].groupby(feature).mean()['SalePrice']# 每一个feature属性的每一个变量的salePrice的平均值
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1,ordering.shape[0]+1)
    #print(ordering)
    ordering = ordering['ordering'].to_dict()
    for cat,o in ordering.items():
        frame.loc[frame[feature] == cat,feature+'_E'] = o
qual_encoded = []
#
# for q in qualitative:
#     encode(train,q)
#     qual_encoded.append(q+'_E')

# 选出了包含缺失数据的行，处理一下
missing_data = all_df.isnull().sum()
missing_data = missing_data[missing_data>0]
ids = all_df[missing_data.index].isnull()
#any() df.any: 接受 0 或 1 来整列或整行判断是否有至少一个 True.
#print(all_df.loc[ids[ids.any(axis=1)].index][missing_data.index])

#相关性计算
def spearman(frame,features):
    spr = pd.DataFrame()
    spr['feature'] = features
    #计算特征和 SalePrice的 斯皮尔曼 相关系数
    spr['spearman'] = [frame[f].corr(frame['SalePrice'],'spearman')for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6,0.25*len(features)))
    sns.barplot(data=spr,y='feature',x='spearman',orient='h')

features = quantitative + qual_encoded
# spearman(train,features)
#plt.show()

# a = train['SalePrice']
# a.plot.hist()
# plt.show()

features = quantitative
standard = train[train['SalePrice'] < np.log(200000)]
pricey = train[train['SalePrice'] >= np.log(200000)]

diff = pd.DataFrame()
diff['feature'] = features
diff['difference'] = [(pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean())/(standard[f].fillna(0.).mean()) for f in features]


sns.barplot(data=diff, x='feature', y='difference')
x=plt.xticks(rotation=90)

features = quantitative + qual_encoded
model = TSNE(n_components=2, random_state=0, perplexity=50)
X = train[features].fillna(0.)








