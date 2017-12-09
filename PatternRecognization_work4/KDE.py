#--coding:utf-8--

from numpy import *
import math
from main import Mat,PrioPro,mu
from KNN import generateDataSet
from sklearn.neighbors import kde

#核密度估计

def getKernelDensityEstimation(data,trainData,h,K):
    M,N=shape(trainData)
    m = M / K
    KDE=zeros(K)
    kde1=kde.KernelDensity(kernel='gaussian',bandwidth=h).fit(trainData[0:m])
    kde2=kde.KernelDensity(kernel='gaussian',bandwidth=h).fit(trainData[m:2*m])
    kde3=kde.KernelDensity(kernel='gaussian',bandwidth=h).fit(trainData[2*m::])
    KDE[0]=exp(kde1.score_samples(data))
    KDE[1]=exp(kde2.score_samples(data))
    KDE[2]=exp(kde3.score_samples(data))
    pdf_KDE=KDE*PrioPro
    print KDE
    return pdf_KDE

def getDistance(vecA,vecB):
    return sqrt(sum(pow(vecA-vecB,2)))

def getMu(data,K):
    M,N=shape(data)
    m=M/K
    mu=zeros((K,N))
    for i in range(K):
        mu[i,:]=mean(data[i*m:(i+1)*m,:],axis=0)
    return mu
'''
def getKernelDensityEstimation(data,trainData,h,K=3):
    mu=getMu(trainData,K)
    #计算高斯核函数
    N=size(data)
    GKF=zeros(K)
    for k in range(K):
        temp=getDistance(data,mu[k,:])
        temp=temp**2/(2*pow(h,2))
        GKF[k]=exp(-temp)
        GKF[k]=GKF[k]/(h ** N)
    PDF=GKF*PrioPro
    return PDF
'''

def getErrorRate(probaDensity,trueLabel):
    m=size(trueLabel)
    predictedLabel=argmax(probaDensity,axis=1)+1
    errorNum=size(nonzero(map(int,trueLabel)-predictedLabel))
    errorRate=float(errorNum)/m
    #print '得到的预测标签为：',predictedLabel
    return errorRate


Iter=10
K=3
setOfH=arange(0.5,10.5,0.5)
sizeOfH=size(setOfH)
errorRate_H=zeros(sizeOfH)
errorRate_Iter=zeros(Iter)
h_index=0
for h in setOfH:
    for iter in range(Iter):
        trainData, trainLabel, testData, testLabel, validationData, validationLabel = generateDataSet(Mat)
        validationData_size = shape(validationData)[0]
        predictedLabelSet_pdf=zeros((validationData_size,K))
        for i in range(validationData_size):
            predictedLabelSet_pdf[i]=getKernelDensityEstimation(validationData[i],trainData,h,K=3)
        errorRate_Iter[iter]=getErrorRate(predictedLabelSet_pdf,validationLabel)
    errorRate_H[h_index]=sum(errorRate_Iter)/Iter
    h_index+=1
print '平滑核函数方法：\n' \
      '通过验证数据集来选择平滑参数的数目H（H的范围为arange(0.5,10.5,0.5)），\n' \
      '在不同的H值下，验证集合对应的平均准确率为（在每个H值下迭代运行10次）：\n' ,1-errorRate_H

minError_H_index=argmin(errorRate_H)
optimal_H=setOfH[minError_H_index]
#选定K后：
Iter=10
errorRate_test=zeros(Iter)
for iter in range(10):
    trainData, trainLabel, testData, testLabel, validationData, validationLabel = generateDataSet(Mat)
    testData_size = shape(testData)[0]
    predictedLabelSet_pdf=zeros((testData_size,K))
    for i in range(testData_size):
        predictedLabelSet_pdf[i] = getKernelDensityEstimation(testData[i], trainData, optimal_H, K)
    errorRate_test[iter]=getErrorRate(predictedLabelSet_pdf,testLabel)
    #print errorRate_test
print '选择平均准确率最高的H：%d 作为模型的参数，再使用测试数据集进行测试，\n' \
      '由于数据集划分的随机性，运行10次取平均值得到的准确率为 %f' % (optimal_H,1-sum(errorRate_test)/Iter)