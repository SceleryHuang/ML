import numpy as np
import operator
from os import listdir



def img2vector(filename):
    returnVect = np.zeros(1024)
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[32*i+j]= np.squeeze(int(lineStr[j]))
    return returnVect

def classify0(inX,dataSet,labels,k):
    '''
    inX :用于分类的输入向量
    dataSet: 输入的训练样本集
    labels:样本数据的类标签向量
    k:用于选择最近邻居的数目
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet

    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis= 1)

    distances = sqDistance**0.5

    sortedDisIndicies = distances.argsort()
    classCount = {}

    for i in range(k):

        votelabel = labels[sortedDisIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel,0) +1

    sortedClassCount = sorted(classCount.items(),key= operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]



def handwritingClassTest():
    # 样本数据的类标签列表
    hwLabels = []

    # 样本数据文件列表
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)

    # 初始化样本数据矩阵（M*1024）
    trainingMat = np.zeros((m,1024))

    # 依次读取所有样本数据到数据矩阵
    for i in range(m):
        # 提取文件名中的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        # 将样本数据存入矩阵
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    # 循环读取测试数据
    testFileList = listdir('digits/testDigits')

    # 初始化错误率
    errorCount = 0.0
    mTest = len(testFileList)

    # 循环测试每个测试数据文件
    for i in range(mTest):
        # 提取文件名中的数字
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        # 提取数据向量
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)

        # 对数据文件进行分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        # 打印KNN算法分类结果和真实的分类
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))

        # 判断KNN算法结果是否准确
        if (classifierResult != classNumStr): errorCount += 1.0

    # 打印错误率
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))



if __name__ =="__main__":
   # testVector = img2vector('C:/User/HXQ/PycharmProjects/ML/digits/testDigits/0_1.txt')
    testVector = img2vector('C:\\Users\\HXQ\\PycharmProjects\\ML\\digits\\testDigits\\0_1.txt')
    print(testVector[0:31])

    handwritingClassTest()