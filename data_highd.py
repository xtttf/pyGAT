import collections
import numpy as np
import os

feature_name = ["loc_x" ,"loc_y" ,"width" ,"height" ,"xVelocity" ,"yVelocity" ,"xAcceleration" ,"yAcceleration" ,"frontSightDistance" ,"backSightDistance" ,"dhw" ,"thw" ,"ttc" ,"precedingXVelocity" ,"precedingId" ,"followingId" ,"leftPrecedingId" ,"leftAlongsideId" ,"leftFollowingId" ,"rightPrecedingId" ,"rightAlongsideId" ,"rightFollowingId" ,"laneID"]
#23个特征
n = 23
feature_out = []
relation = []
num = 1 #节点编号

path = os.getcwd()
data = np.loadtxt(open(os.path.join(path, 'original_data', 'highd', '02_tracks.csv'), "rb"), delimiter=",", skiprows=1)
#delimiter为分隔符，skiprows为跳过前n行，usecols为使用的列数

time_interval = dict()
#记录各车出现的时间段
car_id = data[0][1]
start_time = data[0][0]
all_time = data[0][0]
for i in range(len(data)-1):
    if data[i+1][1] == car_id:
        continue
    end_time = data[i][0]
    all_time = max(all_time, end_time)
    time_interval[car_id] = [start_time, end_time]
    start_time = data[i+1][0]
    car_id = data[i+1][1]

#挑选在interval秒区间内包含车辆最多的情况
interval = 150
car_nums = collections.defaultdict(list)
#car_nums中key为开始时间，value为[开始时间，开始时间+interval-1]区间中存在的车辆id
max_car = 0
start_t = 0
for key, value in time_interval.items():
    if value[1] - value[0] < interval - 1:
        continue
    for i in range(int(value[0]), int(value[1])-interval+2):
        car_nums[i].append(key)
#max_car代表在start_t到start_t+150这段时间内包含的车辆数
for key, value in car_nums.items():
    if len(value) > max_car:
        max_car = len(value)
        start_t = key

#找出挑选后的车辆在数据中的位置
loc = []
k = 0
for i in range(len(data)):
    if data[i][0] == start_t:
        if data[i][1] == car_nums[start_t][k]:
            loc.append(i)
            k += 1
            if k >= len(car_nums[start_t]):
                break

#构造节点
for i in range(interval):
    for j in range(n):
        feature = [num]
        if num-23 > 0:
            relation.append((num-23 ,num))
        num += 1
        for k in range(len(loc)):
            feature.append(data[loc[k]+i][j+2])
        feature.append(feature_name[j])
        feature_out.append(tuple(feature))

np.savetxt(r'data/highD/3.content', np.array(feature_out), encoding='utf-8', fmt='%s')
np.savetxt(r'data/highD/4.cites', np.array(relation), encoding='utf-8', fmt='%s')

f1 = open("data/highD/3.content", "r")
f2 = open("data/highD/4.content", "w")
line = f1.readline()
while line:
    line = line.replace('(', '')
    line = line.replace(',', '')
    line = line.replace(')', '')

    f2.write(line)
    line = f1.readline()
f1.close()
f2.close()






