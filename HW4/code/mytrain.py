import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk

personSize = 41             # 数据集中人数
faceSize = 10               # 每人拥有的图像数
trainSize = 5               # 每人用于训练的图像数
IMAGE_SIZE = (50, 50)       # 将图像统一变化到该大小
pcSize = 200                # 选择多少个特征脸（PCs）
pcSize2 = 2500
outputFace = 10             # 输出的特征脸数目


# 从文件读入图像建立数据集
def createDatabase(path):
    T = []
    # 读取路径下所有的训练图像并
    for j in range(1, personSize+1):
        for i in range(1, trainSize+1):
            # 读入经过预处理的图像
            temp_path=path+'/s'+str(j)+'/'+str(i)+'.pgm'
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
def eigenfaceCore(T, energy):
    # 求平均脸，并将数据集中的数据做0均值化，axis = 1代表对各行求均值
    m = T.mean(axis = 1)
    A = T-m
    L = (A.T) * (A)

    # 输出平均脸
    temp_mean = np.array(m.reshape(IMAGE_SIZE))
    cv.imwrite("mean.jpg", temp_mean)
    # 求协方差矩阵
    LL = np.cov(T, rowvar = 1)
    # LL = np.cov(A, rowvar=1)


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
    # DD = eigenVectors

    # 根据energy的值计算应该取前多少个特征脸
    sum_vv = sum(VV)
    now_sum=0
    PC_index=0
    for i in range(0, VV.size):
        now_sum+=VV[i]
        PC_index = i
        if(1.0*now_sum/sum_vv*100>=energy):
            break
    if(energy==100.0):
        PC_index=VV.size
    pcSize2=PC_index

    # print(energy)
    # print(VV.size)
    VV = eigenValues[0:pcSize2]
    DD = eigenVectors[:, 0:pcSize2]

    # print(VV.shape)
    # print(DD.shape)

    # 拼接、保存并展示前十张特征脸（第二种算法）
    target = Image.new('L', (IMAGE_SIZE[0]*outputFace, IMAGE_SIZE[1]))

    left = 0
    right = IMAGE_SIZE[0]
    # 写出特征脸图像
    for i in range(0,outputFace):
        temp_DD = np.array(DD[:, i].reshape(IMAGE_SIZE))
        dist_DD = cv.normalize(temp_DD, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
        cv.imwrite(str(i)+"_2.jpg",dist_DD)
        target.paste(Image.open(str(i)+"_2.jpg"), (left, 0, right, IMAGE_SIZE[1]))
        left += IMAGE_SIZE[0]
        right += IMAGE_SIZE[0]
    quantity_value = 100
    target.save('EigenFace2.jpg', quantity=quantity_value)
    target_show=cv.imread('EigenFace2.jpg')
    cv.imshow("EigenFace2",target_show)


    L_eig = []
    for i in range(pcSize):
        L_eig.append(D[:,i])

    L_eig = np.mat(np.reshape(np.array(L_eig),(-1,len(L_eig))))
    DD=np.mat(DD)

    # 第一种方法计算特征脸——计算 A * （A^T的特征向量）
    eigenface = A * L_eig


    # 拼接、保存并展示前十张特征脸（第一种算法）
    left = 0
    right = IMAGE_SIZE[0]
    # 写出特征脸图像
    for i in range(0, outputFace):
        temp_eigen = np.array(eigenface[:, i].reshape(IMAGE_SIZE))
        dist_eigen = cv.normalize(temp_eigen, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
        cv.imwrite(str(i) + ".jpg", dist_eigen)
        target.paste(Image.open(str(i) + ".jpg"), (left, 0, right, IMAGE_SIZE[1]))
        left += IMAGE_SIZE[0]
        right += IMAGE_SIZE[0]
    # 图片的质量 0~100
    # quantity_value = 100
    # target.save('EigenFace.jpg', quantity=quantity_value)
    # target_show = cv.imread('EigenFace.jpg')
    # cv.imshow("EigenFace", target_show)

    # 训练结果数据集保存路径
    savepath = tkinter.filedialog.asksaveasfilename()  # 选择以什么文件名保存，返回文件名
    # 保存model文件
    np.savez(savepath, eigenface=eigenface, m=m, A=A, DD=DD)
    print(eigenface.shape)
    print(m.shape)
    print(A.shape)
    print(DD.shape)
    print("Finish training!")


    norm = np.linalg.norm(eigenface, axis=0, keepdims=True)
    # print(norm.shape)
    eigenface = eigenface / norm
    # print(eigenface.T*eigenface)

    return eigenface, m, A, DD


# 选择测试图片
def selectPath(root, btn1, energy):
    # 选择训练集图像所在路径
    datapath = tkinter.filedialog.askdirectory()   # 选择目录，返回目录名
    if datapath != '':
        T = createDatabase(datapath)
        # eigenface,m,A,DD = eigenfaceCore(T, energy)

        # 设置开始训练按钮
        btn1.config(command=lambda: eigenfaceCore(T, energy))
        btn1.config(text="开始训练")
        btn1.pack()

        # # 训练结果数据集保存路径
        # savepath = tkinter.filedialog.asksaveasfilename()     #选择以什么文件名保存，返回文件名
        # # 保存model文件
        # np.savez(savepath, eigenface=eigenface, m=m, A=A, DD=DD)
        # print(eigenface.shape)
        # print(m.shape)
        # print(A.shape)
        # print(DD.shape)
        # print("Finish training!")
    return

# 显示可视化界面
def GUI():
    root = tk.Tk()
    # 能量百分比变量
    v = tk.StringVar()
    # 标题
    root.title("Face-train")
    l = tk.Label(root)      # 显示选择的图像的label
    l.config(text="人脸识别——模型训练演示DEMO")
    l.pack()
    btn1 = tk.Button(root)  # 选择数据集按钮
    # 控制能量值得滑块
    scale = tk.Scale(root)
    scale.config(from_=0, to=100, borderwidth=2, digits=5, orient =tk.HORIZONTAL,label="能量百分比（%）", length=200, variable=v)
    scale.set(100)
    scale.pack()

    # 选择测试图像按钮的设置和显示
    btn = tk.Button(root)
    btn.config(text="选择训练集所在路径")
    btn.config(command=lambda: selectPath(root, btn1, (float)(v.get())))
    btn.pack()
    root.mainloop()


if __name__ == "__main__":
    GUI()
