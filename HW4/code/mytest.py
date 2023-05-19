import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk


personSize = 45         # 数据集中人数
faceSize = 10            # 每人拥有的训练图像数
trainSize = 5
IMAGE_SIZE =(50, 50)     # 将图像统一变化到该大小
pcSize = 200            # 选择多少个特征脸（PCs）
pcSize2 = 2500
outputFace = 50         # 输出的特征脸数目


# 识别人脸
def recognizeFace(filename,databasename):
    testImage=filename
    database=np.load(databasename)
    eigenface=np.mat(database['eigenface'])
    m=np.mat(database['m'])
    A=np.mat(database['A'])
    DD=np.mat(database['DD'])

    # print(eigenface.T*eigenface)
    # _,trainNumber = np.shape(eigenface)     # 特征脸空间向量个数
    # 使用第一种方法计算的特征向量，将人脸集中的图像投影到特征脸空间
    projectedImage = eigenface.T * (A)
    # 第二种方法投影计算的特征向量，将人脸集中的图像投影到特征脸空间
    projectedImage2 = DD.T * (A)
    # 可解决中文路径不能打开问题
    testImageArray = cv.imdecode(np.fromfile(testImage, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    # 转为1-D
    testImageArray=cv.resize(testImageArray, IMAGE_SIZE)
    testImageArray = testImageArray.reshape(testImageArray.size, 1)
    testImageArray = np.mat(np.array(testImageArray))
    differenceTestImage = testImageArray - m
    # 第一种方法投影计算的特征向量，将测试图像投影到特征脸空间
    projectedTestImage = eigenface.T*(differenceTestImage)
    # 第二种方法投影计算的特征向量，将测试图像投影到特征脸空间
    projectedTestImage2 = DD.T * (differenceTestImage)

    # print(trainNumber)
    # print("^^^")
    # print(projectedImage2.shape)
    distance = []
    distance2 = []
    for i in range(0, projectedImage2.shape[1]):
        q = projectedImage[:,i]
        temp = np.linalg.norm(projectedTestImage - q)
        distance.append(temp)
        q2 = projectedImage2[:, i]
        temp2 = np.linalg.norm(projectedTestImage2 - q2)
        distance2.append(temp2)

    minDistance = min(distance)
    index = distance.index(minDistance)
    minDistance2 = min(distance2)
    index2 = distance2.index(minDistance2)

    # 进行人脸识别
    result_index = index2
    # print(index2)
    # 显示人脸库中与该人脸最像的图像
    temp_path='./att_faces'+'/s'+str((int)(result_index/trainSize+1))+'/'+str(result_index%faceSize+1)+'.pgm'
    cv.imshow("recognize result",cv.imread(temp_path))
    cv.waitKey()

    return index2




# 选择测试图片
def selectTest(l,btn1,btn2,root):
    # print(type(energy))
    filename = tkinter.filedialog.askopenfilename()
    # filename = tkinter.filedialog.askdirectory()   # 选择目录，返回目录名
    if filename != '':
        # 读取测试图像显示在图形化界面
        testImg=Image.open(filename)
        tkImg=ImageTk.PhotoImage(testImg)
        l.config(image=tkImg)

        # 设置选择数据集按钮
        btn1.config(command=lambda: selectDatabase(btn2, root, filename))
        btn1.config(text="选择数据集")
        # 显示图像以及选择数据集按钮
        l.pack()
        btn1.pack()
        # 重新绘制
        root.mainloop()


# 选择数据集
def selectDatabase(btn2, root, filename):
    databasename = tkinter.filedialog.askopenfilename()
    if databasename != '':
        # 设置开始重构按钮
        btn2.config(command=lambda: recognizeFace(filename, databasename))
        btn2.config(text="开始识别")
        # 显示开始重构按钮
        btn2.pack()
        # 重新绘制
        root.mainloop()


# 显示可视化界面
def GUI():
    root = tk.Tk()
    # v = tk.StringVar()
    root.title("Face-test")
    l0 = tk.Label(root)  # 标题label
    l0.config(text="人脸识别——识别测试演示DEMO")
    l0.pack()
    l = tk.Label(root)      # 显示选择的图像的label
    btn1 = tk.Button(root)  # 选择数据集按钮
    btn2 = tk.Button(root)  # 开始识别按钮
    # scale = tk.Scale(root)
    # scale.config(from_=0, to=100, borderwidth=1, digits=5, orient =tk.HORIZONTAL,label="能量百分比（%）",variable=v)
    # scale.pack()

    # 选择测试图像按钮的设置和显示
    btn = tk.Button(root)
    btn.config(text="选择测试图片")
    btn.config(command=lambda: selectTest(l, btn1, btn2, root))
    btn.pack()
    root.mainloop()


if __name__ == "__main__":
    GUI()
