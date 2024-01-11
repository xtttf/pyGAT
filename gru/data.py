# coding=utf-8
import random

import numpy as np
import pymongo
import csv
import os

node_limit = 2000  # 限制车数量
# 选取的车辆特征
'''
feature_names = ["acceleration", "allow_speed", "angle", "change_lane", "distance", "lane_id", "lane_position",
                 "last_action_time", "leader_vehicle_distance", "left_follower_vehicle_distance",
                 "left_leader_vehicle_distance", "max_acceleration", "max_deceleration", "max_speed", "position_x",
                 "position_y", "right_follower_vehicle_distance", "right_leader_vehicle_distance", "road_id",
                 "signal_states", "speed", "tua", "wait_time","leader_vehicle_speed" ,"follower_vehicle_distance" ,"follower_vehicle_speed" ,
                 "left_leader_vehicle_speed" ,"left_follower_vehicle_speed" ,"right_leader_vehicle_speed" ,"right_follower_vehicle_speed"]
'''
feature_names = ["acceleration",  "distance", "speed"]

# 需要字符串特殊处理的特征
str_feature_names = ["road_id", "lane_id"]

path = os.getcwd()
file = open(os.path.join(path, 'original_data', 'sumo', 'sparsed_data.csv'), 'r', encoding='UTF8', newline='')


eps = 1e-6

feature_names_set = set(feature_names)
feature_index = {}
for index in range(0, len(feature_names)):
    feature_index[feature_names[index]] = index

str_feature_names_set = set(str_feature_names)

nodes = csv.DictReader(file)

index = 1
index_map = {}
output = []

fea = ["follower_vehicle_speed"]
for node in nodes:
    '''
    #考虑前后车辆因素的特殊处理
    cnt = [0 for _ in range(0, len(feature_names) )]
    for k, v in node.items():
        if k not in feature_names_set:
            continue
        if k in str_feature_names_set:
            if index_map.get(v, None) is None:
                index_map[v] = index
                index += 1
            v = index_map[v]
        if k in fea:
            cnt[feature_index[k]-1] = (v + cnt[feature_index[k]-1])/2
            continue
        cnt[feature_index[k]] = float(v)
    output.append(cnt)
    '''
    
    cnt = [0 for _ in range(0, len(feature_names) + 1)]
    for k, v in node.items():
        if k not in feature_names_set:
            continue
        if k in str_feature_names_set:
            if index_map.get(v, None) is None:
                index_map[v] = index
                index += 1
            v = index_map[v]
        cnt[feature_index[k]] = round(float(v), 2)
    output.append(cnt)

# 制造错误数据

# 修改的数据
modify_feature = "speed"

'''
for i in range(0, len(output)):
    cnt = output[i]
    index = feature_index[modify_feature]
    rand = random.randint(0, 100)
    if rand < 50 or cnt[index] == 0:
        output[i] = cnt
        continue
    else:
        mode = 1
    error_percent = 0
    cnt[len(cnt)-1] = mode
    #构造错误情况
    if mode == 1:
        if random.randint(0, 1):
            error_percent = 10000 + random.randint(501, 100000)
        else:
            error_percent = 10000 - random.randint(501, 10000)

    cnt[index] = cnt[index] * error_percent / 10000.0
    output[i] = cnt
    '''

for i in range(len(output)):
    cnt = output[i]
    index = feature_index[modify_feature]
    index = random.randint(0, 5)
    if index == 0:
        continue
    elif index == 1:
        cnt[index] = 50
    elif index == 2:
        cnt[index] += 20
    elif index == 3:
        cnt[index] = random.randint(0, 50)
    elif index == 4:
        cnt[index] = max(cnt[index]+random.randint(-20, 20), 0)
    else:
        cnt[index] = 0
    output[i] = cnt


np.savetxt(r'gru_data/sparse/selected_lf/train', np.array(output), encoding='utf-8', fmt='%s')
