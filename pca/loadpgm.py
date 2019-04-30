# coding=utf-8
import os
import numpy as np
from PIL import Image

'''
单张图片的读取与图片向矩阵数据的转换
    #p2与p4可以正常读取
    img = Image.open("D:\\pycharm\\project\\pca\\faces\\an2i\\an2i_left_angry_open_2.pgm")
    #颜色变成黑白，本来就是黑白的就省去这一步就行了
    img2=img.convert("L")
    img2.show()
    data=img.getdata()
    data=np.matrix(data)
    print(data)
'''
def loadpgm(filepath):
    first_file=True
    for root, dirs, files in os.walk(filepath):
        #人名
        rootpicture=root.split("\\")[-1]+".jpeg"
        for file in files:
            #print(os.path.splitext(file)[0][-1])
            #文件后缀名
            if first_file:
                if os.path.splitext(file)[0][-23::] == "left_angry_sunglasses_2":
                    img = Image.open((os.path.join(root, file)))
                    oneperson = img.getdata()
                    img.save(rootpicture)
                    oneperson = np.array(oneperson).tolist()
                    first_file=False
            else:
                if os.path.splitext(file)[0][-23::] == "left_angry_sunglasses_2":
                    img = Image.open((os.path.join(root, file)))
                    rootimg = Image.fromarray(np.array(img.getdata())).convert("RGB")
                    rootimg.save(rootpicture)
                    oneperson=oneperson+(np.array(img.getdata()).tolist())
                #img.show()
                #print(img.size)
    return np.array(oneperson)

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





