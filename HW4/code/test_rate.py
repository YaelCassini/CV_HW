import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

personSize = 41
faceSize = 10
trainSize = 5
IMAGE_SIZE =(50,50)
pcSize = 200
pcSize2 = 10
outputFace = 10

# 从文件读入图像建立数据集
def createDatabase(path):
    T = []
    # 读取路径下所有的训练图像并
    for j in range(1, personSize+1):
        for i in range(1, trainSize+1):
            # 读入经过预处理的图像
            temp_path=path+'/s'+str(j)+'/masked_'+str(i)+'.pgm'
            image = cv.imread(temp_path, cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, IMAGE_SIZE)
            # 转为1-D
            image = image.reshape(image.size, 1)
            T.append(image)

    T = np.array(T)
    # 直接reshape会打乱顺序
    T = T.reshape(T.shape[0], T.shape[1])

    return np.mat(T).T

# 计算平均脸、特征脸
def eigenfaceCore(T,pcSize,pcSize2):
    # 求平均脸，并将数据集中的数据做0均值化，axis = 1代表对各行求均值
    m = T.mean(axis = 1)
    A = T-m
    L = (A.T) * (A)
    # 求协方差矩阵
    LL = np.cov(T,rowvar = 1)

    # 第一种方法计算特征值和特征向量矩阵
    eigenValues, eigenVectors = np.linalg.eigh(L)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    V=eigenValues[0:pcSize]
    D=eigenVectors[:,0:pcSize]

    # 计算协方差矩阵的特征值和特征向量
    eigenValues, eigenVectors = np.linalg.eigh(LL)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    VV = eigenValues
    DD = eigenVectors

    L_eig = []
    for i in range(pcSize):
        L_eig.append(D[:,i])

    L_eig = np.mat(np.reshape(np.array(L_eig),(-1,len(L_eig))))
    DD=np.mat(DD)

    # 计算 A *AT的特征向量
    eigenface = A * L_eig
    return eigenface,m,A,DD

# 识别人脸
def recognizeFace(testImage, eigenface,m,A,DD):
    _,trainNumber = np.shape(eigenface)
    # 投影到特征脸后的
    projectedImage = eigenface.T*(A)
    projectedImage2 = DD.T * (A)
    # 可解决中文路径不能打开问题
    testImageArray = cv.imdecode(np.fromfile(testImage,dtype=np.uint8),cv.IMREAD_GRAYSCALE)
    # 转为1-D
    testImageArray=cv.resize(testImageArray,IMAGE_SIZE)
    testImageArray = testImageArray.reshape(testImageArray.size,1)
    testImageArray = np.mat(np.array(testImageArray))
    differenceTestImage = testImageArray - m
    projectedTestImage = eigenface.T*(differenceTestImage)
    projectedTestImage2 = DD.T * (differenceTestImage)

    distance = []
    distance2 = []
    for i in range(0, projectedImage2.shape[1]):
        q = projectedImage[:,i]
        temp = np.linalg.norm(projectedTestImage - q)
        distance.append(temp)
        q2 = projectedImage2[:, i]
        temp2 = np.linalg.norm(projectedTestImage2 - q2)
        distance2.append(temp2)

    # print(projectedImage.shape)
    # print(projectedImage2.shape)


    minDistance = min(distance)
    index = distance.index(minDistance)
    minDistance2 = min(distance2)
    index2 = distance2.index(minDistance2)
    return index, index2


# 进行人脸识别主程序
def mainRecognize(k):
    T = createDatabase('./att_faces')
    count1=0
    count2 = 0
    total=0
    eigenface,m,A,DD = eigenfaceCore(T,k,k)
    for i in range(1,personSize+1):
        for j in range(trainSize+1,faceSize+1):
            testimage = './att_faces' + '/s' + str(i) + '/masked_' + str(
                j) + '.pgm'
            # print(testimage)
            index1,index2 = recognizeFace(testimage, eigenface, m, A, DD)
            temp_path = './att_faces' + '/s' + str((int)(index2 / trainSize + 1)) + '/masked_' + str(
                index2 % trainSize + 1) + '.pgm'
            # print("*****")
            # print(testimage)
            # print(temp_path)
            total=total+1
            if(i==(int)(index1 / trainSize + 1)):
                count1=count1+1
            if (i == (int)(index2 / trainSize + 1)):
                count2 = count2 + 1
            # cv.imshow("recognize result", cv.imread(temp_path, cv.IMREAD_GRAYSCALE))
            # cv.waitKey()
    # print("total")
    print(str(total)+"  "+str(count1)+"  "+str(count2))
    return total,count1,count2


def test_rate():
    T = createDatabase('./att_faces')
    x=[]
    rate1=[]
    rate2=[]
    for k in range(1,5):
        # eigenface,m,A,DD = eigenfaceCore(T,k,k)
        total,count1,count2=mainRecognize(k)
        x.append(k)
        rate1.append(1.0*count1/total)
        rate2.append(1.0 * count2 / total)
    for k in range(1,20):
        # eigenface,m,A,DD = eigenfaceCore(T,k,k)
        total,count1,count2=mainRecognize(k*5)
        x.append(k*5)
        rate1.append(1.0*count1/total)
        rate2.append(1.0 * count2 / total)

    # 设置图表标题，并给坐标轴添加标签
    plt.title("test rate", fontsize=20)
    plt.xlabel("PCsize", fontsize=12)
    plt.ylabel("Rate", fontsize=12)

    # 设置坐标轴刻度标记的大小
    plt.tick_params(axis='both', labelsize=10)

    plt.plot(x, rate1, 'ro-')
    plt.plot(x, rate2, 'g*:', ms=10)
    plt.show()


if __name__ == "__main__":
    test_rate()