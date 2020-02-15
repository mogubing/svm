# -*- coding: utf-8 -*-
_author_ = 'huihui.gong'
_date_ = '2020/1/20'
# svm作为有监督的分类模型（事先给数据打上分类标签，这样我们就知道数据属于哪个分类），可以帮助我们模式识别，分类，回归分析。
# 无监督学习，数据并没有被打上分类标签，可能是我们不具备没有先验的知识，或许打标签的成本很高
# svm分类的过程就是找到一个超平面（这个超平面就是svm分类器），将红球和篮球分开，二维平面变三维空间，原来的曲线变成一个平面我们称这个平面为超平面。。
# ----------------------------------------------------------------------------------------
# 在机器学习领域，总是看到“算法的鲁棒性”这类字眼，比如这句--L1范数比L2范数鲁棒。
# Huber从稳健统计的角度系统地给出了鲁棒性3个层面的概念：
# 模型具有较高的精度或有效性，这也是对于机器学习中所有学习模型的基本要求；
# 对于模型假设出现的较小偏差，只能对算法性能产生较小的影响； 主要是：噪声（noise）   
# 对于模型假设出现的较大偏差，不可对算法性能产生“灾难性”的影响；主要是：离群点（outlier）

from sklearn import svm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
breastData=pd.read_csv('D:\\PycharmProjects\\actualProject1116\\decisiontree200114\\datas\\breast_cancer_detection.csv')
print(breastData.shape)
print(breastData.isnull().any())
breastData=breastData.drop_duplicates()
breastData=breastData.drop(columns='id')
# 不转化成数字，正确率结果也一样
breastData['diagnosis']=breastData['diagnosis'].map({'M':0,'B':1})
print(breastData.shape)
# print(breastData.describe())
features_mean=breastData.iloc[:,1:11]
features_se=breastData.iloc[:,11:21]
features_worst=breastData.iloc[:,21:31]
# 通过绘图查看数据的相关性，做特征选择，特征选择的目的是降维，用少量的特征代表数据的特性，这样也可以增强数据的泛化能力，避免数据过拟合。
# 查看diagnosis列的各值的计数
sns.countplot(breastData['diagnosis'],label='count')
# 查看各变量之间的关系
corr=features_mean.corr()
plt.figure(figsize=(4,14))
# 显示每个方格的数据
sns.heatmap(corr,annot=True)
plt.show()
# 通过热力图可以在相关性强的变量中选取一个变量作为代表参与模型的构建;使用PCA进行特征选择也可以，但是现在这种方法可解释性更强。
x=pd.DataFrame(breastData,columns=['radius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean'])
y=breastData.iloc[:,0]
print(x)
train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.7)
# 数据变换中的数据规范化,让数据在同一个量级上。
ss=StandardScaler()
train_x=ss.fit_transform(train_x,train_y)
test_x=ss.transform(test_x)
print(train_x[:5])
breastSVMclf=svm.SVC(kernel='rbf',C=1.0)
breastSVMclf.fit(train_x,train_y)
predit_y=breastSVMclf.predict(test_x)
print("准确率为：",breastSVMclf.score(test_x,test_y))
print("精确率为:",metrics.precision_score(predit_y,test_y))
print("召回率为:",metrics.recall_score(predit_y,test_y))
# 三个指标皆为百分之90以上，模型训练的还可以。如果不做特征选择，得分为百分之87

