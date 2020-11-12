# -*- coding: utf-8 -*-     支持文件中出现中文字符
###################################################################################################################

"""
Created on Thu Nov 5 20:15:02 2020

@author: Huangjiyuan

代码功能描述: （1）输出各个k取值下的混淆矩阵
            （2）计算各个k取值下的预测准确度

"""
###################################################################################################################

import sys
import numpy as np
import pandas as pd

#1.读取文件并将两个文件合并
dt114 = pd.read_excel('result_114.xlsx')    #读取之前处理得到的result_114文件
dt114 = dt114.iloc[:,1:]
dt114.columns = ['label','mean','var','dwt_appro','dwt_detail','sampen','hurst','pfd']
dt114_label = dt114['label']                #将标签单独存储为一个矩阵

dt_pre = pd.read_excel('result_pd.xlsx')    #读取knn得到的预测结果的文件
dt_pre = dt_pre.iloc[:,1:]

columns_name=['k=1','k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','k=10'] #定义一个列名称list

#2.得到各个k值对应的混淆矩阵
confusion = np.zeros(shape=(3,3))           #定义3x3的混淆矩阵
confusion_pd = pd.DataFrame(data=confusion,columns=['pre_0','pre_1','pre_2'],index=['real_0','real_1','real_2'])    #为混淆矩阵添加列名和行名

for j in range(10):
    for i in range(1,len(dt_pre)+1):
        col = int(format(dt_pre.iloc[-i,j]))    #考虑到测试集是从后往前读取的，因此此处同样从后往前读取
        row = int(format(dt114_label.iloc[-i]))
        confusion_pd.iloc[row,col] += 1         #根据预测值和实际值的情况，在混淆矩阵对应位置加一
    print('k=%d时的混淆矩阵为：\n'%(j+1),confusion_pd)  #将对应k取值情况的混线矩阵的情况打印出来
    confusion_pd[confusion_pd>0] = 0            #将混淆矩阵全部变为0，防止影响之后的计算

#3.得到各个k值对应的预测准确度
t = np.zeros(shape=(len(dt_pre),10),dtype=str)          #定义一个和测试集行数相同，列数为10的，数据类型为str的矩阵
judge_re = pd.DataFrame(data=t,columns=columns_name)    #为矩阵的列添加名称
for i in range(1,len(dt_pre)+1):
    for j in range(10):
        a = dt_pre.iloc[-i,j]    #预测的标签
        b = dt114_label.iloc[-i] #实际的标签
        if a == b:
            judge_re.iloc[-i,j] = 'Yes'
        else:
            judge_re.iloc[-i,j] = 'No'

acc = {}    #定义一个存储准确度的字典
for i in columns_name:
    a = judge_re.groupby([i]).size().reset_index(name='count')  #对judge_re矩阵中的，对应的k取值的列进行统计
    n_num = a.iloc[0,1]     #内容为no的数量
    y_num = a.iloc[1,1]     #内容为yes的数量
    acc[i] = y_num/(n_num+y_num)    #准确度
print(acc)