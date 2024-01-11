# coding=utf-8
import random

import pymongo
import numpy as np

vehicle_limit = 2000     # 限制车数量
# 选取的车辆特征，30个
feature_names = ["speed" ,"max_speed" ,"acceleration" ,"max_acceleration" ,"max_deceleration" ,"allow_speed" ,"position_x" ,"position_y" ,"angle" , "road_id" ,"lane_id" ,"lane_position" ,"change_lane" ,"leader_vehicle_distance" ,"leader_vehicle_speed" ,"follower_vehicle_distance" ,"follower_vehicle_speed" ,"left_leader_vehicle_distance" ,"left_leader_vehicle_speed" ,"left_follower_vehicle_distance" ,"left_follower_vehicle_speed" ,"right_leader_vehicle_distance" ,"right_leader_vehicle_speed" ,"right_follower_vehicle_distance" ,"right_follower_vehicle_speed" ,"distance" ,"signal_states" ,"tua" ,"last_action_time" ,"wait_time"]
# 需要字符串特殊处理的特征
str_feature_names = ["lane_id" ,"road_id"]
mongodb_host = "101.200.42.201"  # mongodb地址
dataset_version = "v0.0.102"  # 数据集版本号

mongo_client = pymongo.MongoClient("mongodb://{}/".format(mongodb_host))
db = mongo_client["sumo_dataset"]
col = db["vehicle-{}".format(dataset_version)]  # 数据集
eps = 1e-6

# 获取列表的第二个元素
def take_second(elem):
    return elem[1]


# 选取部分车辆
vehicle_info = {}
vehicle_index_map = {}
all_vehicle_id = set()
for vehicle in col.find({}, {"vehicle_id": 1, "timestamp": 1}):
    vehicle_id, timestamp = vehicle["vehicle_id"], vehicle["timestamp"]
    all_vehicle_id.add(vehicle_id)
    if not vehicle_info.get(vehicle_id, None):
        vehicle_info[vehicle_id] = {"start_time": 0x3f3f3f3f, "end_time": 0}
    vehicle_info[vehicle_id]["start_time"] = min(vehicle_info[vehicle_id]["start_time"], timestamp)
    vehicle_info[vehicle_id]["end_time"] = max(vehicle_info[vehicle_id]["end_time"], timestamp)
vehicle_sort_tuple, select_vehicle_id = [], []
for vehicle_id in all_vehicle_id:
    vehicle_sort_tuple.append((vehicle_id, vehicle_info[vehicle_id]["end_time"] - vehicle_info[vehicle_id]["start_time"] + 1))
vehicle_sort_tuple.sort(key=take_second, reverse=True)
for index in range(0, min(vehicle_limit, len(vehicle_sort_tuple))):
    select_vehicle_id.append(vehicle_sort_tuple[index][0])
for index in range(len(select_vehicle_id)):
    vehicle_index_map[select_vehicle_id[index]] = index

tot_index = 1
index_map = {}
nodes = {}
feature_names_set = set(feature_names)
str_feature_names_set = set(str_feature_names)
start_time, end_time = 0x3f3f3f3f, 0
for vehicle_node in col.find({"vehicle_id": {"$in": select_vehicle_id}}):
    timestamp = vehicle_node['timestamp']
    vehicle_index = vehicle_index_map[vehicle_node["vehicle_id"]]
    start_time = min(start_time, timestamp)
    end_time = max(end_time, timestamp)
    for k, v in vehicle_node.items():
        if k not in feature_names_set:
            continue
        if k in str_feature_names_set:
            if not index_map.get(v, None):
                index_map[v] = tot_index
                tot_index += 1
            v = index_map[v]    # 转成数字
        key = "{}_{}".format(k, timestamp)
        if not index_map.get(key, None):
            index_map[key] = tot_index
            tot_index += 1
        key_index = index_map[key]
        if not nodes.get(key, None):
            feature = [0 for _ in range(len(select_vehicle_id)+1)]
            feature[0] = key_index
            feature.append(k)
            nodes[key] = feature
        
        nodes[key][vehicle_index+1] = v
       
        

select_node = {}
features_output = []
for key, node in nodes.items():
    _sum = 0
    for index in range(1, len(node)-1):
        _sum += node[index]
    if -eps < _sum < eps:
        continue
    select_node[key] = True

    features_output.append(tuple(node))

# 随机打乱顺序
n = len(features_output)
for i in range(0, n-1):
    j = random.randint(i, n-1)
    tmp = features_output[i]
    features_output[i] = features_output[j]
    features_output[j] = tmp
np.savetxt(r'data/xtf/xtf3.content', np.array(features_output), encoding='utf-8', fmt='%s')

adj_output = []
for feature_name in feature_names:
    keys = []
    for timestamp in range(start_time, end_time+1):
        key = "{}_{}".format(feature_name, timestamp)
        if not select_node.get(key, False):
            continue
        keys.append(index_map[key])
    for index in range(0, len(keys)-1):
        adj_output.append((keys[index], keys[index+1]))

# for timestamp in range(start_time, end_time+1):
#     keys = []
#     for feature_name in feature_names:
#         key = "{}_{}".format(feature_name, timestamp)
#         if not select_node.get(key, False):
#             continue
#         keys.append(index_map[key])
#     for i in range(0, len(keys)):
#         for j in range(i+1, len(keys)):
#             adj_output.append((keys[i], keys[j]))
np.savetxt(r'data/xtf/xtf3.cites', np.array(adj_output), encoding='utf-8', fmt='%s')
f = open("data/xtf/README3", "w")
f.write("start_time: {} end_time: {}\n".format(start_time, end_time))
f.write("node num: {}\n".format(len(features_output)))
f.write("vehicle num: {}\n".format(len(select_vehicle_id)))
f.write("features num: {}\n".format(len(feature_names)))
f.write("features index map:\n")
for k, v in select_node.items():
    f.write("{}: {}\n".format(k, index_map[k]))
f.close()
