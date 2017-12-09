# -- coding:utf-8 --
from numpy import *
from main import Mat
import operator
'''
def generateRand(lower,upper):
    counter=(upper-lower)+1
    randList=[]
    count=0
    while count<counter:
        randIndex=random.randint(lower,upper)
        if randIndex not in randList:
            randList.append(randIndex)
            count=count+1
            #print randList
    return randList
'''
def normalizationData(Mat):
    M,N=shape(Mat)
    data=Mat[:,0:N-1]
    min_data=data.min(0)
    max_data=data.max(0)
    data=(data-min_data)/(max_data-min_data)
    Mat[:,0:N-1]=data
    return Mat

#将数据划分为8:1:1比例的训练数据，测试数据与验证数据
def generateDataSet(Mat):
    M,N=shape(Mat)
    data=Mat[:,0:N-1]
    label=Mat[:,N-1]
    train_size=int(0.8*M/3);test_size=int(0.1*M/3);validation_size=int(0.1*M/3)
    test_index=[]
    validation_index=[]
    train_index=[]
    for i in range(3):
        train_random_index=range(i*50,(i+1)*50)
        random.shuffle(train_random_index)
        #print train_random_index,shape(train_random_index)
        test_index.extend(train_random_index[0:test_size])
        validation_index.extend(train_random_index[test_size:(test_size+validation_size)])
        train_index.extend(train_random_index[test_size+validation_size::])
    trainData=data[train_index];trainLabel=label[train_index]
    testData=data[test_index];testLabel=label[test_index]
    validationData=data[validation_index];validationLabel=label[validation_index]
    return trainData,trainLabel,testData,testLabel,validationData,validationLabel

def getDistance(vecA,vecB):
    return sqrt(sum(pow(vecA-vecB,2),axis=1))

def NearestNeighbor(data,vec,label,numOfNeighbors):
    M,N=shape(data)
    dist=getDistance(tile(vec,(M,1)),data)
    distSorted=argsort(dist)
    classCount={}
    for num in range(numOfNeighbors):
        voteLabel=label[distSorted[num]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    #print classCount.items(),'\n'
    predictedLabel=sortedClassCount[0][0]
    #print predictedLabel
    return predictedLabel
'''
#最近邻估计方法
def getNearestNeighborErrorRate(Mat,numOfNeighbors):
    M,N=shape(Mat)
    label=Mat[:,N-1]
    trainData=Mat[:,0:N-1]
    predictedLabelSet=zeros(M)
    for i in range(M):
        predictedLabelSet[i]=NearestNeighbor(Mat,trainData[i],numOfNeighbors)
    print 'KNN得到的预测标签为：',predictedLabelSet
    errorNum=size(nonzero(label-predictedLabelSet))
    errorRate=errorNum/float(M)
    return errorRate
'''
def getErrorRate(predictedLabel,Label):
    m=size(Label)
    errorNum=size(nonzero(Label-predictedLabel))
    errorRate=float(errorNum)/m
    return errorRate

#最近邻决策方法
print '最近邻决策方法:'
Iter=10
errorRate_K=zeros(5)
errorRate_Iter=zeros(Iter)
Mat=normalizationData(Mat)
for K in range(1,6):
    for iter in range(Iter):
        trainData, trainLabel, testData, testLabel, validationData, validationLabel = generateDataSet(Mat)
        validationData_size = shape(validationData)[0]
        predictedLabelSet_valid=zeros(validationData_size)
        for i in range(validationData_size):
            predictedLabelSet_valid[i]=NearestNeighbor(trainData,validationData[i],trainLabel,K)
        errorRate_Iter[iter]=getErrorRate(predictedLabelSet_valid,validationLabel)
    errorRate_K[K-1]=sum(errorRate_Iter)/Iter
print '通过验证数据集来选择近邻的数目K（K的范围为1至5），\n' \
      '在不同的K值下，验证集合对应的平均准确率为（在每个K值下迭代运行10次）：' ,1-errorRate_K

minError_K_index=argmin(errorRate_K)
optimal_K=minError_K_index+1
print '由验证集合选择出错误率最小的K为：',optimal_K
Iter=10
testData_size=shape(testData)[0]
errorRate_test=zeros(Iter)
for iter in range(10):
    trainData, trainLabel, testData, testLabel, validationData, validationLabel = generateDataSet(Mat)
    predictedLabelSet_test=zeros(testData_size)
    for i in range(testData_size):
        predictedLabelSet_test[i]=NearestNeighbor(trainData,testData[i],trainLabel,optimal_K)
    errorRate_test[iter]=getErrorRate(predictedLabelSet_test,testLabel)
    #print errorRate_test
print '选择平均准确率最高的H作为模型的参数通过测试数据集进行测试，由于数据集划分的随机性，' \
      '运行10次取平均值得到的准确率为',1-sum(errorRate_test)/Iter
print '\n'