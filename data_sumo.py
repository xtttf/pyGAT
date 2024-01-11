import collections
import numpy as np
import os
import csv

# 选取的车辆特征，30个
title = ["vehicle_id", "timestamp"]
feature_name = ["speed", "max_speed", "acceleration", "max_acceleration", "max_deceleration", "allow_speed", "position_x", "position_y", "angle",
                "road_id", "lane_id", "lane_position", "change_lane", "leader_vehicle_distance", "leader_vehicle_speed", "follower_vehicle_distance",
                "follower_vehicle_speed", "left_leader_vehicle_distance", "left_leader_vehicle_speed", "left_follower_vehicle_distance", "left_follower_vehicle_speed",
                "right_leader_vehicle_distance", "right_leader_vehicle_speed", "right_follower_vehicle_distance", "right_follower_vehicle_speed", "distance",
                "signal_states", "tua", "last_action_time", "wait_time"]
# 需要字符串特殊处理的特征
str_feature_names = ["lane_id", "road_id"]
str_feature = set(str_feature_names)
n = 30
feature_out = []
relation = []



path = os.getcwd()
filename = os.path.join(path, 'original_data', 'sumo', 'normal_data.csv')

tot_index = 1
index_map = {}
data = []

with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for line in reader:
        data_t = []
        for k, v in line.items():
            #对不全是数字的特征进行修改
            if k in str_feature:
                if not index_map.get(v, None):
                    index_map[v] = tot_index
                    tot_index += 1
                v = index_map[v]
            if k in title or k in feature_name:
                data_t.append(float(v))
        data.append(data_t)


time_interval = dict()
#记录各车出现的时间段
car_id = data[0][0]
start_time = data[0][1]
end_time = data[0][1]
for i in range(len(data)):
    if data[i][0] not in time_interval:
        start_time = data[i][1]
        end_time = data[i][1]
        time_interval[data[i][0]] = [start_time, end_time]
    else:
        time_interval[data[i][0]][1] = data[i][1]


#挑选在interval秒区间内包含车辆最多的情况
#interval根据数据情况进行调整
interval = 65
car_nums = collections.defaultdict(list)
#car_nums中key为开始时间，value为[开始时间，开始时间+interval-1]区间中存在的车辆id
max_car = 0
start_t = 0
for key, value in time_interval.items():
    if value[1] - value[0] < interval - 1:
        continue
    for i in range(int(value[0]), int(value[1])-interval+2):
        car_nums[i].append(key)
for key, value in car_nums.items():
    if len(value) > max_car:
        max_car = len(value)
        start_t = key

#查找对应车辆在数据中的位置
#loc[i][j]代表第i时刻，在car_nums[start_t]中第j个车辆的数据位置
loc = [[0 for _ in range(max_car)] for _ in range(interval)]
car_num = car_nums[start_t]
car = {}
for i in range(len(car_num)):
    car[car_num[i]] = i
for i in range(len(data)):
    if data[i][0] in car and data[i][1] in range(start_t, start_t+interval):
        loc[int(data[i][1]-start_t)][car[data[i][0]]] = i

#构造节点
#构造边可以修改，目前构造的边是将t时刻与t-1时刻的同一数据特征相连，可以后期改为同一时刻的所有数据特征相连
num = 1 #节点编号
for i in range(interval):
    for j in range(n):
        feature = [num]#输出中第一个为节点编号

        for k in range(max_car):
            ll = loc[i][k]
            feature.append(data[ll][j+2])

        # 构造图的边，将t时刻与t-1时刻的同一特征相连
        if num-n > 0:
            relation.append((num-n, num))

        '''
        # 构造图的边，将同一时刻的所有特征相连
        if j%n>0:
            for e in range(j%n):
                relation.append((num-e-1, num))
        '''
        # 仅将speed与同一时刻的其他节点相连
        if num%30 == 1:
            for e in range(1, 30):
                relation.append((num, num+e))

        feature.append(feature_name[j])#最后加上特征名
        feature_out.append(tuple(feature))
        num += 1


np.savetxt(r'data/sumo/t.content', np.array(feature_out), encoding='utf-8', fmt='%s')
np.savetxt(r'data/sumo/test.cites', np.array(relation), encoding='utf-8', fmt='%s')

f1 = open("data/sumo/t.content", "r")
f2 = open("data/sumo/test.content", "w")
line = f1.readline()
while line:
    line = line.replace('(', '')
    line = line.replace(',', '')
    line = line.replace(')', '')

    f2.write(line)
    line = f1.readline()
f1.close()
f2.close()






