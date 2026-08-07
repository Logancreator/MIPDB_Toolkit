import codecs
import json
import math
import os
from glob import glob

# 根据坐标转换矩形框
def getRectangles(json_path, offset):
    info = []

    data = json.load(open(json_path, encoding='UTF-8'))
    width = data['imageWidth']
    height = data['imageHeight']
    for line in data['shapes']:


        left_x, left_y, right_x, right_y = 0, 0, 0, 0
        for index, point in enumerate(line['points']):

            point = [item / 4 for item in point]
            if left_x == 0:
                left_x = point[0]
                right_x = point[0]
                left_y = point[1]
                right_y = point[1]
            else:
                if point[0] < left_x:
                    left_x = point[0]
                if point[0] > right_x:
                    right_x = point[0]
                if point[1] < left_y:
                    left_y = point[1]
                if point[1] > right_y:
                    right_y = point[1]
        info.append(
            {
                'label': ('leaf' if 'y' in line['label'] else 'root'),
                'bbox': [
                    math.floor(left_x) - offset if math.floor(left_x) - offset > 0 else 0,
                    math.floor(left_y) - offset if math.floor(left_y) - offset > 0 else 0,
                    math.ceil(right_x) + offset if math.ceil(right_x) + offset < width else width,
                    math.ceil(right_y) + offset if math.ceil(right_y) + offset < height else height
                ]
            }
        )
    return info

# 批量读取json并转换成xml文件
def readJsons(jsons_path, xmls_save_path):
    '''

    :param jsons_path: 存放所有json文件的文件夹路径
    :param xmls_save_path: 转换成xml格式的保存路径
    '''
    files = glob(jsons_path + "\*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]

    for file_name in files:
        json_path = os.path.join(jsons_path, file_name + ".json")
        infos = getRectangles(json_path, offset=20)
        with codecs.open(xmls_save_path + "Annotations/" + file_name + ".xml", "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'cornDataset' + '</folder>\n')
            xml.write('\t<filename>' + file_name + ".jpg" + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>cornDataset</database>\n')
            xml.write('\t\t<annotation>corn</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t</source>\n')
            # xml.write('\t<size>\n')
            # xml.write('\t\t<width>' + str(width) + '</width>\n')
            # xml.write('\t\t<height>' + str(height) + '</height>\n')
            # xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
            # xml.write('\t</size>\n')
            xml.write('\t<segmented>0</segmented>\n')
            for index, info in enumerate(infos):
                name = info['label']
                rectangle = info['bbox']

                xmin = rectangle[0]
                ymin = rectangle[1]
                xmax = rectangle[2]
                ymax = rectangle[3]

                if xmax <= xmin:
                    print('xmax={a}, xmin = {b}不标注该目标'.format(a=xmax, b=xmin))
                elif ymax <= ymin:
                    print('ymax={a}, ymin = {b}不标注该目标'.format(a=ymax, b=ymin))
                else:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + name + '</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    # if xmin <= 0 or xmax >= float(width) or ymin <= 0 or ymax >= float(height):
                    #     xml.write('\t\t<truncated>1</truncated>\n')
                    # else:
                    #     xml.write('\t\t<truncated>0</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
            xml.write('</annotation>')

readJsons(jsons_path='xxx', xmls_save_path='xxx')