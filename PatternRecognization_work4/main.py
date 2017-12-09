# -- coding:utf-8 --
from numpy import *
import csv
import math
K=3#数据类别数
PrioPro=array([1./3,1./3,1./3])#数据的先验概率

#加载数据
def loadData():
    Mat=[]
    with open('HWData3.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            line=[float(x) for x in line]
            Mat.append(line)
        Mat=array(Mat)
    return Mat


#获取数据的均值与协方差矩阵
def getParameters(Mat,K):
    m,n=shape(Mat)
    mu=zeros((K,n-1))
    sigma=zeros((K,n-1,n-1))
    for k in range(K):
        data=array([ x for x in Mat if x[n-1]==k+1 ])
        mu[k]=mean(data[:,0:n-1],axis=0)
        sigma[k]=cov(data[:,0:n-1],rowvar=False)
    return mu,sigma
#似然率测试规则
def getLikelihoodRate(K,Mat,sigma,mu,PrioPro):
    m,n=shape(Mat)
    data=Mat[:,0:n-1]
    Px_w = mat(zeros((m, K)))
    for i in range(K):
        coef = (2 * math.pi) ** (-(n-1) / 2.) * (linalg.det(sigma[i]) ** (-0.5))
        temp=multiply((data-mu[i])*mat(sigma[i]).I,data-mu[i])
        Xshift = sum(temp, axis=1)
        Px_w[:,i]= coef * exp(Xshift*-0.5)  #矩阵与常数相乘
    likelihoodRate=mat(zeros((m,K)))
    for i in range(K):
        likelihoodRate[:,i]=PrioPro[i]*Px_w[:,i]
    likelihoodRate=array(likelihoodRate)
    return likelihoodRate

#根据概率计算错误率
def getErrorRate(probaDensity,Mat):
    m,n=shape(Mat)
    predictedLabel=argmax(probaDensity,axis=1)+1
    label=Mat[:,n-1]
    errorNum=size(nonzero(map(int,label)-predictedLabel))
    errorRate=float(errorNum)/m
    print '得到的预测标签为：',predictedLabel
    return errorRate

#朴素贝叶斯估计方法
def NaiveBayes(Mat,K,mu,sigma,PrioPro):
    M,D=shape(Mat)
    data=Mat[:,0:D-1]
    Px_w=ones((M,K))
    for k in range(K):
        Px_d=zeros((M,D-1))
        for d in range(D-1):
            cof=1./(math.sqrt(2*math.pi)*math.sqrt(sigma[k,d,d]))
            coe=(data[:,d]-repeat(mu[k,d],M))**2/(2*sigma[k,d,d])
            Px_d[:,d]=cof*exp(-coe)
            Px_w[:,k]=Px_w[:,k]*Px_d[:,d]
    NaiveBayesPro=Px_w*PrioPro
    #print '朴素贝叶斯方法得出的每类概率密度函数为：',NaiveBayesPro
    return NaiveBayesPro


Mat=loadData()
M,N=shape(Mat)
mu,sigma=getParameters(Mat,K)
print '三类数据的均值分别为：\n',mu
print '三类数据的协方差矩阵分别为：\n',sigma
#似然率测试规则
print '似然率测试规则方法：'
likelihoodRate=getLikelihoodRate(K,Mat,sigma,mu,PrioPro)
errorRate_Likelihood=getErrorRate(likelihoodRate,Mat)
print '由似然率测试规则得出的分类正确率为：',(1-errorRate_Likelihood)
print '\n'

print '朴素贝叶斯分类方法：'
#A,B,C=getKernelDensityEstimation(Mat)
#朴素贝叶斯规则
NaiveBayesPro=NaiveBayes(Mat,K,mu,sigma,PrioPro)
errorRate_NaiveBayes=getErrorRate(NaiveBayesPro,Mat)
print '由朴素贝叶斯方法得出的分正确率为：',(1-errorRate_NaiveBayes)
print '\n'
