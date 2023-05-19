import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk


personSize = 41         # 数据集中人数
faceSize = 10            # 每人拥有的训练图像数
IMAGE_SIZE =(50, 50)     # 将图像统一变化到该大小
SHOW_SIZE =(70, 80)     # 图像展示大小
pcSize = 200            # 选择多少个特征脸（PCs）
pcSize2 = 1000
restructSize = 1000      # 重构使用的特征脸个数
# outputFace = 50         # 输出的特征脸数目


# 重构人脸
def reconstructFace(filename,databasename, restructPC):

    restructSize = int(restructPC)
    # restructSize = 10
    # 读入数据集
    database=np.load(databasename)
    eigenface=np.mat(database['eigenface'])
    m=np.mat(database['m'])
    DD=np.mat(database['DD'])

    # norm = np.linalg.norm(eigenface, axis=0, keepdims=True)
    # # print(norm.shape)
    # eigenface = eigenface / norm
    # print(eigenface.T * eigenface)


    # 特征脸空间向量个数
    # _,trainNumber = np.shape(eigenface)
    print(eigenface.shape)
    print(DD.shape)
    # if(eigenface.shape[1]<restructSize):
    #     print("对不起！没有足够的特征脸！")
    #     return
    #
    eigenface=eigenface[:,0:restructSize]
    DD = DD[:, 0:restructSize]

    # 可解决中文路径不能打开问题
    testImageArray = cv.imdecode(np.fromfile(filename, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    # 将测试图像转为1-D
    testImageArray = cv.resize(testImageArray, IMAGE_SIZE)
    testImageArray = testImageArray.reshape(testImageArray.size,  1)
    testImageArray = np.mat(np.array(testImageArray))
    # differenceTestImage = testImageArray + m
    differenceTestImage = testImageArray

    # 投影到特征脸空间
    projectedTestImage = DD.T * (differenceTestImage)
    projectedTestImage2 = eigenface.T * (differenceTestImage)

    # 重构
    rebuild = DD * projectedTestImage
    rebuild2 = eigenface * projectedTestImage2
    # 将计算结果归一化到0-255
    rebuild = cv.normalize(rebuild, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    # rebuild = rebuild - m
    rebuild = rebuild
    rebuild = np.array(rebuild.reshape(IMAGE_SIZE))

    # rebuild2 = cv.normalize(rebuild2, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    # # rebuild2 = rebuild2 - m
    # rebuild2 = rebuild2
    # rebuild2 = np.array(rebuild2.reshape(IMAGE_SIZE))

    # 将计算结果归一化到0-255
    result_rebuild = cv.normalize(rebuild, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    result_rebuild=cv.resize(result_rebuild, SHOW_SIZE)
    cv.imshow("result_rebuild", result_rebuild)
    cv.imwrite("reconstruct.jpg", result_rebuild)

    # result_rebuild2 = cv.normalize(rebuild2, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    # result_rebuild2 = cv.resize(result_rebuild2, SHOW_SIZE)
    # cv.imshow("result_rebuild2", result_rebuild2)
    # cv.imwrite("reconstruct2.jpg", result_rebuild2)
    # print(eigenface.T*eigenface)
    return


# 选择测试图片
def selectTest(l,btn1,btn2,root, restructPC):
    filename = tkinter.filedialog.askopenfilename()
    # filename = tkinter.filedialog.askdirectory()   # 选择目录，返回目录名
    if filename != '':
        # 读取测试图像显示在图形化界面
        testImg=Image.open(filename)
        tkImg=ImageTk.PhotoImage(testImg)
        l.config(image=tkImg)

        # 设置选择数据集按钮
        btn1.config(command=lambda: selectDatabase(btn2, root, filename, restructPC))
        btn1.config(text="选择数据集")
        # 显示图像以及选择数据集按钮
        l.pack()
        btn1.pack()
        # 重新绘制
        root.mainloop()


# 选择数据集
def selectDatabase(btn2, root, filename, restructPC):
    databasename = tkinter.filedialog.askopenfilename()
    if databasename != '':
        # 设置开始重构按钮
        btn2.config(command=lambda: reconstructFace(filename, databasename, restructPC))
        btn2.config(text="开始重构")
        # 显示开始重构按钮
        btn2.pack()
        # 重新绘制
        root.mainloop()


# 显示可视化界面
def GUI():
    root = tk.Tk()
    v = tk.StringVar()
    root.title("Face-reconstruct")
    l0 = tk.Label(root)  # 标题label
    l0.config(text="人脸识别——人脸重构演示DEMO")
    l0.pack()
    l = tk.Label(root)      # 显示选择的图像的label
    btn1 = tk.Button(root)  # 选择数据集按钮
    btn2 = tk.Button(root)  # 开始识别按钮
    scale = tk.Scale(root)
    scale.config(from_=0, to=2500, borderwidth=2,orient =tk.HORIZONTAL,label="重构使用特征脸数目",variable=v, length=200)
    scale.set(500)
    scale.pack()

    # 选择测试图像按钮的设置和显示
    btn = tk.Button(root)
    btn.config(text="选择测试图片")
    btn.config(command=lambda: selectTest(l, btn1, btn2, root, (float)(v.get())))
    btn.pack()
    root.mainloop()


if __name__ == "__main__":
    GUI()
    # filename="D:\STUDY\CV\\att_faces\s41\\7.pgm"
    # databasename = "D:\STUDY\CV\\888.npz"
    # restructPC = 2500
    # reconstructFace(filename, databasename, restructPC)
