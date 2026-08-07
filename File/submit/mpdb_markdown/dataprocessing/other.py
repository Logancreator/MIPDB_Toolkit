from PIL import ImageColor
from xml.dom.minidom import parse
from sklearn.model_selection import train_test_split
import cv2

import base64
import json
import os
from glob import glob

from PIL import Image, ImageOps
# 检查转换后的xml文件，画框
def checkLabel(jpg_path, xml_path):
    STANDARD_COLORS = [
        'black', 'DarkOrange', 'IndianRed', 'FireBrick', 'HotPink', 'Gold', 'AliceBlue', 'Salmon', 'Chartreuse', 'Aqua',
        'Aquamarine', 'Azure', 'Beige', 'Bisque',
        'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki',
        'DarkOrchid', 'DarkSalmon', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FloralWhite',
        'Fuchsia', 'Gainsboro', 'GhostWhite', 'GoldenRod',
        'Tan', 'HoneyDew', 'HotPink', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightPink', 'LightSalmon', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow',
        'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSlateBlue',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'SandyBrown',
        'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SteelBlue',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow',
    ]
    dom = parse(xml_path)
    objects = dom.getElementsByTagName("object")
    colors = [ImageColor.getrgb(STANDARD_COLORS[i % len(STANDARD_COLORS)]) for i in range(0, len(objects))]
    img = cv2.imread(jpg_path)

    for index, object in enumerate(objects):
        xmin = object.getElementsByTagName('xmin')[0].childNodes[0].data
        ymin = object.getElementsByTagName('ymin')[0].childNodes[0].data
        xmax = object.getElementsByTagName('xmax')[0].childNodes[0].data
        ymax = object.getElementsByTagName('ymax')[0].childNodes[0].data
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[index], 1)
    cv2.imwrite('result.jpg', img)

# 统计每张图片的目标个数
def getObjectNums():
    total_files = glob(r"G:\jsons\*.json")
    for jsonFile in total_files:
        data = json.load(open(jsonFile, encoding='UTF-8'))
        if len(data['shapes']) >= 255:
            print(jsonFile)
            print(len(data['shapes']))

# 拆分训练集、测试集、验证集
def splitTrainAndVal(isUseTest, saved_path):
    '''

    :param isUseTest: 是否划分测试集
    :param saved_path: 结果保存路径
    :return:
    '''
    txtsavepath = saved_path + "/ImageSets/Main/"
    ftrainval = open(txtsavepath + '/trainval.txt', 'w')
    ftest = open(txtsavepath + '/test.txt', 'w')
    ftrain = open(txtsavepath + '/train.txt', 'w')
    fval = open(txtsavepath + '/val.txt', 'w')
    total_files = glob(saved_path + "/Annotations/" + '*.xml')
    total_files = [i.replace("\\", "/").split("/")[-1].split(".xml")[0] for i in total_files]
    trainval_files = []
    test_files = []
    if isUseTest:
        trainval_files, test_files = train_test_split(total_files, test_size=0.002, random_state=55)
    else:
        trainval_files = total_files
    for file in trainval_files:
        ftrainval.write(file + "\n")
    train_files, val_files = train_test_split(trainval_files, test_size=0.10, random_state=55)
    # train
    for file in train_files:
        ftrain.write(file + "\n")
    # val
    for file in val_files:
        fval.write(file + "\n")
    for file in test_files:
        print(file)
        ftest.write(file + "\n")
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


# 根据json['imgaeData']生成原图
def jsonToJpg(jsons_path, save_path, resize_ratio=None):

    total_files = os.listdir(jsons_path)
    print(len(total_files))
    for index, jsonFile in enumerate(total_files):
        print(index + 1)
        jsonPath = os.path.join(jsons_path, jsonFile)

        imageData = json.load(open(jsonPath, encoding='UTF-8'))['imageData']
        imageData = base64.b64decode(imageData)  # 解码

        jpgPath = os.path.join(save_path, jsonFile.split('.')[0] + '.JPG')
        file = open(jpgPath, "wb")
        file.write(imageData)
        file.close()
        if resize_ratio != None:

            image = Image.open(os.path.join(save_path, jsonFile.split('.')[0] + '.JPG'))
            image = ImageOps.exif_transpose(image)
            image = image.resize((int(image.size[0] / resize_ratio), int(image.size[1] / resize_ratio)))
            image.save(os.path.join(save_path, jsonFile.split('.')[0] + '.JPG'))
