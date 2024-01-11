import random
import csv
import numpy as np
import os

path = os.path.dirname(__file__)

feature_name = ["x" ,"y" ,"width" ,"height" ,"xVelocity" ,"yVelocity" ,"xAcceleration" ,"yAcceleration" ,"frontSightDistance" ,"backSightDistance" ,"dhw" ,"thw" ,"ttc" ,
                "precedingXVelocity" ,"precedingId" ,"followingId" ,"leftPrecedingId" ,"leftAlongsideId" ,"leftFollowingId" ,"rightPrecedingId" ,"rightAlongsideId" ,
                "rightFollowingId" ,"laneId"]
#23个特征
feature_name_selected = ["backSightDistance" ,"x", "rightFollowingId" ,"ttc" ,"xVelocity", "xAcceleration", "yAcceleration"]
csvfile = open("C:\\Users\\biubiubiu\\Desktop\\csv\\02_tracks.csv" ,"r")
file = csv.DictReader(csvfile)
data = [row for row in file]
feature_change = "xVelocity"
error_type = 5

#错误数据的五种攻击形式
def error_data(type, feature):
    if type == 1:
        return 50
    elif type == 2:
        return float(feature)+15
    elif type == 3:
        return random.randint(0, 50)
    elif type == 4:
        return float(feature)+random.randint(-20,20)
    else:
        return 0

#生成数据
def generate_data(left ,right ,rate ,feature_in ):
    feature_out = []
    for i in range(left ,right):
        f = []
        rand = random.randint(0, 100)
        if rand < rate:
            for j in range(len(feature_in)):
                f.append(float(data[i][feature_in[j]]))
            f.append(0)
        else:
            for j in range(len(feature_in)):
                if feature_in[j] == feature_change:

                    index = error_data(error_type, data[i][feature_in[j]])
                    f.append(index)
                else:
                    f.append(float(data[i][feature_in[j]]))
            f.append(1)
        feature_out.append(f)
    return feature_out

#生成包含上一时刻速度特征的数据集
def previous_included_data(left, right, rate, feature_in):
    feature_out = []
    for i in range(left ,right):
        f = []
        rand = random.randint(0, 100)
        if rand < rate and int(data[i]['frame']) != 1:
            for j in range(len(feature_in)):
                f.append(float(data[i][feature_in[j]]))
            f.append(data[i-1][feature_change])
            f.append(0)
        if rand >= rate and int(data[i]['frame']) != 1:
            for j in range(len(feature_in)):
                if feature_in[j] == feature_change:
                    #生成随机错误数据

                    index = error_data(error_type, data[i][feature_in[j]])
                    f.append(index)
                else:
                    f.append(float(data[i][feature_in[j]]))
            f.append(data[i - 1][feature_change])
            f.append(1)
        if f != []:
            feature_out.append(f)
    return feature_out


break_point = len(data)//2

#生成训练集，正负数据量1:1,错误数据标签为1
data_train_out = generate_data(0 ,break_point ,50 ,feature_name)
data_train_out_selected = generate_data(0 ,break_point ,50 ,feature_name_selected)
data_train_out_previous = previous_included_data(0 ,break_point ,50 ,feature_name_selected)

#生成测试集，正负数据量9：1
data_test_out = generate_data(break_point ,len(data) ,90 ,feature_name)
data_test_out_selected = generate_data(break_point ,len(data) ,90 ,feature_name_selected)
data_test_out_previous = previous_included_data(break_point ,len(data) ,90 ,feature_name_selected)

'''
np.savetxt(os.path.join(path, 'gru_data', 'highd', 'attack5', 'data_train'), np.array(data_train_out), encoding='utf-8', fmt='%s')
np.savetxt(os.path.join(path, 'gru_data', 'highd', 'attack5', 'data_train_select'), np.array(data_train_out_selected), encoding = 'utf-8' ,fmt = '%s')
np.savetxt(os.path.join(path, 'gru_data', 'highd', 'attack5', 'data_test'), np.array(data_test_out), encoding = 'utf-8' ,fmt = '%s')
np.savetxt(os.path.join(path, 'gru_data', 'highd', 'attack5', 'data_test_select'), np.array(data_test_out_selected), encoding = 'utf-8' ,fmt = '%s')
'''
np.savetxt(os.path.join(path, 'gru_data', 'highd', 'attack5', 'data_train_select2'), np.array(data_train_out_previous), encoding='utf-8', fmt='%s')
np.savetxt(os.path.join(path, 'gru_data', 'highd', 'attack5', 'data_test_select2'), np.array(data_test_out_previous), encoding='utf-8', fmt='%s')


