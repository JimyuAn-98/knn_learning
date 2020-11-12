# -*- coding: utf-8 -*-     支持文件中出现中文字符
###################################################################################################################

"""
Created on Thu Nov 5 20:15:02 2020

@author: Huangjiyuan

代码功能描述: （1）读取处理后的数据
            （2）使用KNN完成分类的运算

"""
###################################################################################################################

import pandas as pd
import numpy as np
import math

#0.定义将文件保存的函数
def save_in_xlsx(data,name):    
    writer = pd.ExcelWriter(r'%s.xlsx'%(name))
    data.to_excel(writer,'page_1',float_format='%.6f')
    writer.save()
    writer.close()

#1.读取文件并将两个文件合并
columns_name = ['mean','var','dwt_appro','dwt_detail','sampen','hurst','pfd']               #设置由各个属性组成的矩阵

dt113 = pd.read_excel(r'result_%d.xlsx'% (113))                                             #读取之前处理得到的result_113文件
dt113 = dt113.iloc[:,1:]                                                                    #去除掉第一列，也就是表格中的序列列
dt113.columns = ['label','mean','var','dwt_appro','dwt_detail','sampen','hurst','pfd']      #为每一列添加名称

dt114 = pd.read_excel(r'result_%d.xlsx'% (114))                                             #读取之前处理得到的result_114文件
dt114 = dt114.iloc[:,1:]
dt114.columns = ['label','mean','var','dwt_appro','dwt_detail','sampen','hurst','pfd']

dt = pd.concat([dt113,dt114],axis=0,ignore_index=True)                                      #将两个矩阵合并成一个矩阵

#2.构建KNN
test_data_num = int(len(dt)*0.2) + 1    #定义测试集的数据的量
train_data_num = int(len(dt)*0.8)       #定义训练集的数据的量
label = np.array(dt['label'])           #得到只有标签的矩阵
result = np.zeros(shape=(test_data_num,10))     #定义结果的输出矩阵，有10列，分别对应k从1取到10时的预测值

for i in range(1,test_data_num):                #在全体数据中提取后20%为测试集
    a1 = dt.iloc[-i,:]                          #从后往前提取行数据
    temp0 = np.zeros(shape=(train_data_num,7))  #定义一个矩阵，用来存储计算得到，每一个属性的，差的平方的值
    for j in range(train_data_num):             #根据训练集进行循环，计算两行数据之间差的平方的值
        a2 = dt.iloc[j,:]                       #从前往后取行数据
        for n in range(len(columns_name)):      #根据属性的名称进行循环提取
            n1 = a1[columns_name[n]]            #提取测试集的第i行的第n列
            n2 = a2[columns_name[n]]            #提取训练集的第j行的第n列
            temp0[j][n] = (n1 - n2)**2          #表示测试集与训练集中的第j行计算得到的结果
    
    temp1 = np.zeros(shape = (len(temp0),2))    #定义一个矩阵，用来存储标签值和所有属性的差的平方的和
    for x in range(len(temp0)):
        temp1[x][0] = label[x]                  #矩阵的第一列用于存储标签
        for y in range(len(temp0[0])):          #矩阵的第二列用于存储temp0中所有值的和
            temp1[x][1] += temp0[x][y]
    distance_x = pd.DataFrame([temp1[i][0], math.sqrt(temp1[i][1])] for i in range(len(temp1)))  #对计算得到的每一个差的平方和求开方
    distance_x.columns = ['label','value']      #为每一列添加名称
    
    for k in range(1,11):                           #将k循环取值，得到在不同的k的取值下的预测值
        z1 = distance_x.sort_values(by=['value'])   #根据第二列，也就是欧氏距离，排序
        for m in range(k):                          #取得距离为前k个的值
            zt0 = zt1 = zt2 = 0                     #定义三个用于存储测试集的标签值的变量
            if z1.iloc[m,0] == 0:
                zt0 += 1/(z1.iloc[m,1])**2          #考虑到距离的因素，为添加的值加上权重
            elif z1.iloc[m,0] == 1:
                zt1 += 1/(z1.iloc[m,1])**2
            else:
                zt2 += 1/(z1.iloc[m,1])**2
        if max(zt0,zt1,zt2) == zt0:                 #得到最终的预测标签值
            result[-i][k - 1] = 0                   #同样的从后往前存储
        elif max(zt0,zt1,zt2) == zt1:
            result[-i][k - 1] = 1
        else:
            result[-i][k - 1] = 2
result_pd = pd.DataFrame(result,columns=['k=1','k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','k=10']) #为每一列添加名称
save_in_xlsx(result_pd,'result_pd') #将得到的预测结果的矩阵存储在xlsx文件中