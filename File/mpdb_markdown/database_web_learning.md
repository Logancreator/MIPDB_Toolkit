# <font color=Green>Database and web learning</font>

#### <font color=Gray>By Wenpeng Deng/Jianye Chang/Linsen Dong</font>

## <font color=lightgreen>1.LiveLessons Python Fundamentals</font>
```
廖雪峰python学习网站
https://www.liaoxuefeng.com/wiki/1016959663602400
```
```
python学习手册
https://docs.pythontab.com/
```
```
bilibili【Python教程】《零基础入门学习Python》
https://www.bilibili.com/video/BV1c4411e77t/?spm_id_from=333.337.search-card.all.click&vd_source=c79f6b57cc3412fa64e6ca5ea2e0285c
```
## <font color=lightgreen>2.Mongodb</font>
***一个基于分布式文件存储的数据库***

```https://docs.mongoing.com/```

MongoDB是一个文档数据库，旨在简化开发和扩展。它分许多版本，其中有Python，Node.js，C，shell,C++,JAVA，Ruby。在这里，我们因为编程语言偏好性问题，我们选择Python版本，即pyMongodb进行学习及分享。

### <font color=lightgreen>2.1 文档数据库</font>
MongoDB中的记录是一个文档，它是由字段和值对组成的数据结构。
MongoDB文档类似于JSON对象。
字段的值可以包括其他文档，数组和文档数组。
下面我给出一个例子：
<figure align=center>
    <img src="database_web_learning\1686794007908.jpg" >
</figure>
很明显上述例子是一个类似JSON格式的字典，其中value可以为字符，数字甚至是列表！

### 2.2 主要特性
1.高性能
2.查询语言丰富
3.高可用
4.水平拓展
5.支持多种存储引擎

### 2.3 pyMongodb安装
```
python -m pip install pymongo
```
### 2.4 连接到MongoDB Atlas
Atlas是一个新手友好的MongoDB云数据库
```
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Replace the placeholder with your Atlas connection string
uri = "<connection string>"

# Set the Stable API version when creating a new client
client = MongoClient(uri, server_api=ServerApi('1'))
                          
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
```





## <font color=lightgreen>3.Fastapi</font>
***一种现代，快速，高性能的 Web 框架***

```https://fastapi.tiangolo.com/zh/```
## <font color=lightgreen>4.Vue3+TS</font>
***一个逼格满满的前端技术***

```https://cn.vuejs.org/guide/typescript/overview.html```
## <font color=lightgreen>5.Arco Design</font>
***一个字节跳动推出的组件库***

```https://arco.design/react/docs/start```
## <font color=lightgreen>6.Fetch</font>
***一个用于JavaScript脚本里面发出HTTP请求的工具***

```https://developer.mozilla.org/zh-CN/docs/Web/API/Fetch_API/Using_Fetch```