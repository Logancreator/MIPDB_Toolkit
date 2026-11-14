import codecs
import math
from glob import glob
import json
import os
import cv2


from itertools import groupby
# 根据两点坐标获取线段上所有点坐标
def getLineIndex(x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    if y0 < y1:
        ystep = 1
    else:
        ystep = -1
    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0
    line = []
    for x in range(round(x0), round(x1 + 1)):
        if steep:
            line.append((round(y), round(x)))
        else:
            line.append((round(x), round(y)))

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    return line

def getIndexList(json_path):
    indexList = []
    data = json.load(open(json_path, encoding='UTF-8'))
    for line in data['shapes']:
        index_dict = {}
        pointList = []
        index_dict['label'] = line['label']

        length = len(line['points'])
        for index, point in enumerate(line['points']):
            if index < length - 1:
                tempList = getLineIndex(point[0], point[1], line['points'][index+1][0], line['points'][index+1][1])
                for p in tempList:
                    pointList.append(p)
        index_dict['pointList'] = pointList
        indexList.append(index_dict)
    return indexList


# 等间距抽取10个点作为关键点标注
def selectPoints(indexList):

    for i, line in enumerate(indexList):
        resultList = []
        allPoints = line['pointList']
        step = len(allPoints) // 9
        for index in range(0, len(allPoints), step):
            resultList.append(allPoints[index])
        # last = allPoints[-1]
        # pos = allPoints.index((3043, 549))
        if len(resultList) == 10:
            if resultList[-1] != allPoints[-1]:
                resultList[-1] = allPoints[-1]
        else:
            resultList.append(allPoints[-1])

        assert len(resultList) == 10, print('关键点选取错误！')
        indexList[i]['pointList'] = resultList
    return indexList

# 获取矩形框+关键点
def getRecsAndKeyPoints(img_path, json_path):
    height, width, channels = cv2.imread(img_path).shape
    height = height * 2
    width = width * 2
    info = []
    data = json.load(open(json_path, encoding='UTF-8'))
    # plants = {}
    for line in data['shapes']:

        linePoints = []
        allPoints = []
        length = len(line['points'])
        num_list = [''.join(list(g)) for k, g in groupby(line['label'], key=lambda x: x.isdigit())]

        if num_list[0].isdigit():
            num_plant = int(num_list[0])
        elif num_list[1].isdigit():
            num_plant = int(num_list[1])
        else:
            raise ValueError('wrong num')

        # if num_plant not in plants.keys():
        #     plants[num_plant] = []
        # for point in line['points']:
        #     plants[num_plant].append(point)

        left_x, left_y, right_x, right_y = 0, 0, 0, 0
        for index, point in enumerate(line['points']):
            linePoints.append([round(point[0] / 4), round(point[1] / 4)])
            if index < length - 1:

                tempList = getLineIndex(point[0] / 4,
                                        point[1] / 4,
                                        line['points'][index+1][0] / 4,
                                        line['points'][index+1][1] / 4)
                for p in tempList:
                    allPoints.append(p)

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

        step = len(allPoints) // 9
        keyPoints = []
        if step == 0:
            for i in range(0, len(allPoints)):
                keyPoints.append(allPoints[i][0])
                keyPoints.append(allPoints[i][1])
        else:
            for i in range(0, 9 * step, step):
                keyPoints.append(allPoints[i][0])
                keyPoints.append(allPoints[i][1])
        if len(keyPoints) == 20:
            if keyPoints[-1] != allPoints[-1][1] or keyPoints[-2] != allPoints[-1][0]:
                keyPoints[-1] = allPoints[-1][1]
                keyPoints[-2] = allPoints[-1][0]
        else:
            if len(keyPoints) == 18:
                keyPoints.append(allPoints[-1][0])
                keyPoints.append(allPoints[-1][1])
            else:
                for _ in range(0, 20 - len(keyPoints)):
                    keyPoints.append(0)
        if len(keyPoints) != 20:
            print('error')
        assert len(keyPoints) == 20, print('关键点选取错误！')
        info.append(
            {
                'label': ('leaf' if 'y' in line['label'] else 'root'),
                'bbox': [
                            math.floor(left_x) - 20 if math.floor(left_x) - 20 > 0 else 0,
                            math.floor(left_y) - 20 if math.floor(left_y) - 20 > 0 else 0,
                            math.ceil(right_x) + 20 if math.ceil(right_x) + 20 < width else width,
                            math.ceil(right_y) + 20 if math.ceil(right_y) + 20 < height else height
                         ],
                'keyPoints': keyPoints,
                'linePoints': linePoints
            })

    # for num, plant in plants.items():
    #     left_x, left_y, right_x, right_y = 0, 0, 0, 0
    #     for point in plant:
    #         point = [item / 4 for item in point]
    #         if left_x == 0:
    #             left_x = point[0]
    #             right_x = point[0]
    #             left_y = point[1]
    #             right_y = point[1]
    #         else:
    #             if point[0] < left_x:
    #                 left_x = point[0]
    #             if point[0] > right_x:
    #                 right_x = point[0]
    #             if point[1] < left_y:
    #                 left_y = point[1]
    #             if point[1] > right_y:
    #                 right_y = point[1]
    #     info.append(
    #         {
    #             'label': 'plant',
    #             'bbox': [
    #                 math.floor(left_x) - 20 if math.floor(left_x) - 20 > 0 else 0,
    #                 math.floor(left_y) - 20 if math.floor(left_y) - 20 > 0 else 0,
    #                 math.ceil(right_x) + 20 if math.ceil(right_x) + 20 < width else width,
    #                 math.ceil(right_y) + 20 if math.ceil(right_y) + 20 < height else height
    #             ]
    #         })

    return info



def makeAnnotations(imgs_path, jsons_path, saved_path):

    if not os.path.exists(saved_path + "Annotations"):
        os.makedirs(saved_path + "Annotations")

    files = glob(jsons_path + "\*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]

    # 读取标注信息并写入 xml
    for file_name in files:
        if os.path.exists(saved_path + "Annotations/" + file_name + ".xml"):
            continue
        jpg_path = os.path.join(imgs_path, file_name + '.JPG')
        json_path = os.path.join(jsons_path, file_name + ".json")
        if not os.path.exists(jpg_path):
            print('pass {}'.format(os.path.basename(jpg_path)))
            continue
        print(os.path.basename(jpg_path))
        infos = getRecsAndKeyPoints(jpg_path, json_path)
        height, width, channels = cv2.imread(jpg_path).shape
        height = height * 2
        width = width * 2

        with codecs.open(saved_path + "Annotations/" + file_name + ".xml", "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'cornDataset' + '</folder>\n')
            xml.write('\t<filename>' + file_name + ".jpg" + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>cornDataset</database>\n')
            xml.write('\t\t<annotation>corn</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t</source>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t<segmented>0</segmented>\n')
            for index, info in enumerate(infos):
                name = info['label']
                coordinate = info['bbox']
                if 'keyPoints' in info.keys():
                    assert 'linePoints' in info.keys(), 'key error'
                    keyPoints = info['keyPoints']
                    linePoints = info['linePoints']
                else:
                    keyPoints = None
                    linePoints = None
                xmin = coordinate[0]
                ymin = coordinate[1]
                xmax = coordinate[2]
                ymax = coordinate[3]
                if xmax <= xmin:
                    print('xmax={a}, xmin = {b}不标注该目标'.format(a=xmax, b=xmin))
                elif ymax <= ymin:
                    print('ymax={a}, ymin = {b}不标注该目标'.format(a=ymax, b=ymin))
                else:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + name +'</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    if xmin <= 0 or xmax >= float(width) or ymin <= 0 or ymax >= float(height):
                        xml.write('\t\t<truncated>1</truncated>\n')
                    else:
                        xml.write('\t\t<truncated>0</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    if keyPoints != None and linePoints != None:
                        xml.write('\t\t<keypoints>' + str(keyPoints) +'</keypoints>\n')
                        xml.write('\t\t<linepoints>' + str(linePoints) +'</linepoints>\n')
                    xml.write('\t</object>\n')
            xml.write('</annotation>')


makeAnnotations(imgs_path='xxx', jsons_path='xxx', saved_path='xxx')