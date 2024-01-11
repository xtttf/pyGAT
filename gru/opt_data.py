# coding=utf-8
import random

import numpy as np
import pymongo

node_limit = 2000  # 限制车数量
# 选取的车辆特征
feature_names = ["acceleration", "speed", "lane_position", "distance"]
# 需要字符串特殊处理的特征
mongodb_host = "47.110.46.1:27017"  # mongodb地址
dataset_version = "v0.0.2"  # 数据集版本号

mongo_client = pymongo.MongoClient("mongodb://{}/".format(mongodb_host))
db = mongo_client["sumo_dataset"]
col = db["vehicle-{}".format(dataset_version)]  # 数据集
feature_index_map = {
    "speed": 2,
    "acceleration": 3,
    "lane_position": 4,
    "distance": 5,
}
features = set(feature_names)

eps = 1e-6
nodes = []
speeds = {}

for node in col.find().sort("timestamp", 1).limit(node_limit):
    nodes.append(node)
    key = "{}_{}".format(node["vehicle_id"], node["timestamp"])
    speeds[key] = node["speed"]


result = []
for node in nodes:
    vehicle_id = node["vehicle_id"]
    timestamp = node["timestamp"]
    key = "{}_{}".format(vehicle_id, timestamp)
    cnt = [0 for _ in range(len(features)+3)]
    cnt[0] = speeds.get("{}_{}".format(vehicle_id, timestamp - 2), 0)
    cnt[1] = speeds.get("{}_{}".format(vehicle_id, timestamp - 1), 0)
    # 前序时间速度
    for k, v in node.items():
        if k not in features:
            continue
        cnt[feature_index_map[k]] = v
    result.append(cnt)
# 修改的数据
for index in range(len(result)):
    cnt = result[index]
    cnt.append(cnt[2])
    if random.randint(0, 100) < 90 or cnt[2] == 0:
        result[index] = cnt
        continue
    cnt[len(cnt) - 2] = 1
    if random.randint(0, 1):
        error_percent = 10000 + random.randint(501, 100000)
    else:
        error_percent = 10000 - random.randint(501, 10000)
    cnt[2] = cnt[2] * error_percent / 10000.0
    result[index] = cnt
np.savetxt(r'opt_test', np.array(result), encoding='utf-8', fmt='%s')


