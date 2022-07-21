import pandas as pd
import numpy as np
import datetime
#不知道为什么相对地址突然没法使用了，20220708
data = pd.read_excel(r"D:\0_Nutcloud\Manuscripts\P4 Latex and Code\v1\Preprocessing_part\AFG_GTD_data\globalterrorismdb_0522dist.xlsx")
data = data[data['country_txt'] == "Afghanistan"]

data.count()
data_select = data[data['iyear']>=2005]
data_select = data_select[data_select['iday']!=0]
prov = data_select['provstate'].unique().tolist()
###list按照首字母排序 https://www.cnblogs.com/Anson-Z/p/6825406.html
prov=sorted(prov,key=str.lower) ###将省份A-Z排序，与Adj_distance相统一

datenum = 365*16 + 4 ####Afg 2005-1-1后的数据，4个闰年
Afg_data = np.zeros([36,datenum])###Afg 2005-1-1后的数据，3个闰年
print(Afg_data.shape)
d1 = datetime.datetime.strptime('2005-1-1','%Y-%m-%d')
for i,itor in data_select.iterrows():
    d2 = datetime.datetime.strptime(str(itor['iyear'])+'-'+str(itor['imonth'])+'-'+str(itor['iday']),'%Y-%m-%d')
    d = d2 - d1
    Afg_data[prov.index(itor['provstate']), int(d.days)]=1
    print(i)

date_arrow = ['Date'] ###时间索引
for j in range(datenum): ###制作时间索引，用以拼接
    d_mid = d1 + datetime.timedelta(days=j) ### https://blog.csdn.net/lilongsy/article/details/80242427
    ###date_time转换为str https://www.jquery-az.com/python-datetime-to-string/
    d_mid = d_mid.strftime('%Y/%m/%d')
    date_arrow.append(d_mid)
    
Afg_data = np.column_stack((prov,Afg_data))
Afg_data = np.row_stack((date_arrow,Afg_data))###完整的包含省份、日期、unknown和Kagida province的结果

pure_AFG_data  = Afg_data ###无索引，无Unknown等的数据

for m in range(len(pure_AFG_data)):
    if pure_AFG_data[m][0]=='Unknown':
        pure_AFG_data = np.delete(pure_AFG_data, m, axis=0) ###numpy根据索引删除行 https://www.jb51.net/article/139764.htm
        break
    
for m in range(len(pure_AFG_data)):
    if pure_AFG_data[m][0]=='Paktika Province':###这玩意仅一次
        pure_AFG_data = np.delete(pure_AFG_data, m, axis=0)
        break

pure_AFG_data = pure_AFG_data[1:,1:]

###将数据转置后保存 https://www.jb51.net/article/155472.htm
pure_AFG_data = np.transpose(pure_AFG_data)
Afg_data = np.transpose(Afg_data)

np.savetxt('Afg_data.csv', pure_AFG_data, fmt='%s', delimiter=',')###无任何索引数据，顺序为省份A-Z
np.savetxt('Afg_data_original.csv', Afg_data, fmt='%s', delimiter=',')###含省份、日期索引数据，作为对照