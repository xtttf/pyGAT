# coding=utf-8
import random
import numpy as np
import pymongo

node_limit = 2000  # 限制车数量
# 需要字符串特殊处理的特征
str_feature_names = ["road_id", "lane_id"]
mongodb_host = "47.110.46.1:27017"  # mongodb地址
dataset_version = "v0.0.2"  # 数据集版本号

mongo_client = pymongo.MongoClient("mongodb://{}/".format(mongodb_host))
db = mongo_client["sumo_dataset"]
col = db["vehicle-{}".format(dataset_version)]  # 数据集
eps = 1e-6

result = []
features = {}
for node in col.find().sort("timestamp", 1).limit(node_limit):
    vehicle_id = node["vehicle_id"]
    feature = features.get(vehicle_id, [])
    feature.append((node["speed"], node["timestamp"]))
    features[vehicle_id] = feature


def take_second(elem):
    return elem[1]


for vehicle_id, feature in features.items():
    feature.sort(key=take_second)
    cnt = []
    for item in feature:
        cnt.append([item[0], 0])
    result.append(cnt)

for i in range(0, len(result)):
    cnt = result[i]
    for j in range(0, len(cnt)):
        rand = random.randint(0, 100)
        if rand < 90:
            continue
        cnt[j][1] = 1
        if -eps < cnt[j][0] < eps:
            cnt[j][0] = 1.0
        if random.randint(0, 1):
            error_percent = 10000 + random.randint(501, 100000)
        else:
            error_percent = 10000 - random.randint(501, 10000)
        cnt[j][0] = cnt[j][0] * error_percent / 10000.0
    res = [len(cnt)] + cnt
    result[i] = tuple(res)

print("num: {}".format(len(result)))
np.savetxt(r'dataset', np.array(result), encoding='utf-8', fmt='%s')