# coding=utf-8
import os
import numpy as np
from loadpgm import loadpgm
from PIL import Image

def readfile(path):
    file = open(path)
    next(file)
    first_else = True
    for data in file.readlines():
        data = data.strip("\n")
        nums = data.split("\t")[0:3]
        if first_else:
            matrix = np.array(nums)
            first_else = False
        else:
            matrix = np.c_[matrix, nums]
    file.close()
    # 矩阵为n*m，m为样本数
    return np.mat(matrix).astype(float)


def zeroScale(dataset):
    # 对数据集按行求平均
    meanVals = np.mean(dataset, axis=1)
    # 中心化
    zerodata = dataset - meanVals
    return zerodata


def eigvalPc(eigVals, precentages):
    # 特征值从大到小排序
    sortArray = np.sort(eigVals)[::-1]
    # 百分比
    arrSum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum/arrSum >= precentages:
            ##返回最后的特征值下标
            return num


def pca(dataset, precentages):
    meanVals = np.mean(dataset, axis=1)
    # 中心化
    zerodata = zeroScale(dataset)
    # 计算协方差矩阵
    covMat = np.cov(zerodata)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    k = eigvalPc(eigVals, precentages)
    # 获取对应的特征值排好序的下标
    eigIndex = np.argsort(eigVals)[::-1]
    # 获取对应的特征向量下标
    eigIndex = eigIndex[0:k + 1]
    # 获取主成分特征向量
    eigPCA = eigVects[:, eigIndex]
    # 原始数据投影到低纬数据
    lowDimdata = np.transpose(zerodata) * eigPCA
    #重构数据
    reconMat=(lowDimdata*np.transpose(eigPCA))+np.transpose(meanVals)
    #np.savetxt(outfile, reconMat, delimiter="\t")
    return  reconMat


if __name__ == "__main__":
    for root, dirs, files in os.walk("D:\\pycharm\\project\\pca\\faces\\"):
        first_person = True
        for dir in dirs:
            filepath = (os.path.join(root, dir))
            if first_person:
                dataset = loadpgm(filepath)
                first_person = False
            else:
                dataset = np.column_stack((dataset, loadpgm(filepath)))
    #数据矩阵化
    dataset=np.mat(dataset)
    out=pca(dataset,0.7)
    #10张图片10*3840每张图片为60X64
    for i in range(0,10):
        outpicture=str(i)+".jpeg"
        picture1 = (out[i].reshape(60, 64))
        # 保存重构图片数据
        picture1 = np.real(picture1)
        picture1 = Image.fromarray(picture1).convert("RGB")
        picture1.save(outpicture)


