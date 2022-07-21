### 本文件最后希望得到34*34的相关性Adj矩阵
### spearman-pearson相关性区别，适用性：https://zhuanlan.zhihu.com/p/60059869
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = np.genfromtxt('34_province_yearbook_data.csv', dtype = str, delimiter = ','
                     , skip_header = 1)

x = data[:, 3:].astype(float)###前三列为索引
x = np.transpose(x) ###转置，从34*n(n为特征数)，变为n*34，计算34个省份相似度

### 不同量纲数据归一化
scaler = MinMaxScaler(feature_range=(0, 1)) ###放缩为0-1 range，参考图像处理
x = scaler.fit_transform(x)
    
    
x = pd.DataFrame(x) ###转为pandas dataframe 用pandas算相关性
pearson_matrix = x.corr(method="pearson") ###pandas计算相关性

### 画相关性热力图 https://blog.csdn.net/qq_42898981/article/details/102836530 
plt.subplots(figsize=(9, 9))
#sns.heatmap(pearson_matrix, annot=True, vmax=1, square=True, cmap="Blues")
sns.heatmap(pearson_matrix, cmap="Blues")
plt.show()

pearson_matrix = pearson_matrix.to_numpy()
pearson_matrix_pure = pearson_matrix ###34*34无索引的Adj
pearson_matrix_original = np.column_stack((data[:,:3],pearson_matrix)) ###含索引的


np.savetxt('Adj_similarity.csv', pearson_matrix_pure, fmt='%s', delimiter=',')###无索引Adj
np.savetxt('Adj_similarity_original.csv', pearson_matrix_original, fmt='%s', delimiter=',')###包含索引