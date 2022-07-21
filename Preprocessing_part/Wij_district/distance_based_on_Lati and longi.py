### https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude#43211266
### 本文件最后希望得到34*34的距离矩阵，KM单位
import geopy.distance
import numpy as np
import math

### https://blog.csdn.net/hawkuu/article/details/103425110 对数组每个元素操作，以函数的方式
def Thresholded_Guassian_kernel(x, std): #x:distance里的每个元素 std：distance的方差
    return (math.exp(-(pow(x,2)/pow(std,2))))

#######################################################################################
data = np.genfromtxt('34_province_Lati_longti.csv', dtype = str, delimiter = ','
                     , skip_header = 1)

x = data[:, -2:]###最后两列为纬度、经度

province_list = data[:, :-2]###保留前面的内容
for i in range(len(x)):
    distance_list = [] ###保留x[i]计算的34个距离值
    for j in range(len(x)):
        coords_1 = x[i]
        coords_2 = x[j]
        distance_1_2 = geopy.distance.distance(coords_1, coords_2).km
        distance_list.append(distance_1_2)
    province_list = np.column_stack((province_list, distance_list)) ###包含省份名称index的距离矩阵

#print('The distance matrix is:', province_list[:, 3:])
distance_matrix = province_list[:, 3:]
distance_matrix = distance_matrix.astype(float) ###距离矩阵
std = np.std(distance_matrix) ###计算距离矩阵标准差
distance_matrix_standard = distance_matrix ###高斯门限处理后的标准距离矩阵

for i in range(len(distance_matrix)):
    for j in range(len(distance_matrix[i])):
        if (i!=j): ###高斯门限要求i != j, wij 才有效
            distance_matrix_standard[i][j] = Thresholded_Guassian_kernel(distance_matrix[i][j], std)

print('The pure distance matrix is:', distance_matrix_standard)
np.savetxt('Adj_distance.csv', distance_matrix_standard, fmt='%s', delimiter=',')###无索引的高斯门限Adj
np.savetxt('Adj_distance_original.csv', province_list, fmt='%s', delimiter=',')###包含索引的原始距离