from pymongo import MongoClient
import  datetime
#创建mongodb实例 方式1
#client = MongoClient()

#创建mongodb实例 方式2
client = MongoClient('localhost',8888)

#创建mongodb实例 方式3
#client = MongoClient('mongodb://localhost:27017/')

#获取数据库
db = client.test_database

db = client['test-database']

#获取集合(相当于表)
collection = db.test_collection

collection = db['test_collection']

#创建一个json样式的文档
post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": datetime.datetime.utcnow()}

#插入文档
posts = db.posts
post_id = posts.insert_one(post).inserted_id
print(post_id)