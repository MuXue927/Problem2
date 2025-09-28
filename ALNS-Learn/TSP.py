import copy
import math
import time
import random
import numpy as np
import pandas as pd


# 计算TSP总距离
def dis_cal(path, dist_mat):
    distance = 0
    for i in range(len(path) - 1):
        distance += dist_mat[path[i]][path[i + 1]]
    distance += dist_mat[path[-1]][path[0]]
    return distance

# 随机删除N个城市
def random_destroy(x, destroy_city_cnt):
    new_x = copy.deepcopy(x)
    removed_cities = []

    # 随机选择N个城市, 并删除
    removed_index = random.sample(range(0, len(x)), destroy_city_cnt)
    for i in removed_index:
        removed_cities.append(new_x[i])
        x.remove(new_x[i])
    return removed_cities

# 删除距离最大的N个城市
def max_n_destroy(x, destroy_city_cnt):
    new_x = copy.deepcopy(x)
    removed_cities = []

    # 计算顺序距离并排序
    dis = []
    for i in range(len(new_x) - 1):
        dis.append(dist_mat[new_x[i]][new_x[i + 1]])
    dis.append(dist_mat[new_x[-1]][new_x[0]])
    sorted_index = np.argsort(np.array(dis))

    # 删除距离最大的N个城市
    for i in range(destroy_city_cnt):
        removed_cities.append(new_x[sorted_index[-1 - i]])
        x.remove(new_x[sorted_index[-1 - i]])
    return removed_cities

# 随机删除连续的N个城市
def consecutive_n_destroy(x, destroy_city_cnt):
    new_x = copy.deepcopy(x)
    removed_cities = []

    # 随机选择N个城市, 并删除
    removed_index = random.sample(range(0, len(x) - destroy_city_cnt), 1)[0]
    for i in range(removed_index + destroy_city_cnt, removed_index, -1):
        removed_cities.append(new_x[i])
        x.remove(new_x[i])
    return removed_cities

# destroy操作
def destroy(flag, x, destroy_city_cnt):
    # 三个destroy算子, 第一个是随机删除N个城市, 第二个是删除距离最大的N个城市, 第三个是随机删除连续的N个城市
    removed_cities = []
    if flag == 0:
        # 随机删除N个城市
        removed_cities = random_destroy(x, destroy_city_cnt)
    elif flag == 1:
        # 删除距离最大的N个城市
        removed_cities = max_n_destroy(x, destroy_city_cnt)
    elif flag == 2:
        # 随机删除连续的N个城市
        removed_cities = consecutive_n_destroy(x, destroy_city_cnt)
    return removed_cities

# 随机插入
def random_insert(x, removed_cities):
    insert_index = random.sample(range(0, len(x)), len(removed_cities))
    for i in range(len(removed_cities)):
        x.insert(insert_index[i], removed_cities[i])

# 贪心插入
def greedy_insert(x, removed_cities):
    dis = float('inf')
    insert_index = -1

    for i in range(len(removed_cities)):
        # 寻找插入后的最小距离
        for j in range(len(x) + 1):
            new_x = copy.deepcopy(x)
            new_x.insert(j, removed_cities[i])
            if dis_cal(new_x, dist_mat) < dis:
                dis = dis_cal(new_x, dist_mat)
                insert_index = j

        # 最小位置插入
        x.insert(insert_index, removed_cities[i])
        dis = float('inf')

# repair操作
def repair(flag, x, removed_cities):
    # 两个repair算子, 第一个是随机插入, 第二个是贪心插入
    if flag == 0:
        # 随机插入
        random_insert(x, removed_cities)
    elif flag == 1:
        # 贪心插入
        greedy_insert(x, removed_cities)

# 选择destroy算子
def select_and_destroy(destroy_w, x, destroy_city_cnt):
    # 轮盘赌选择destroy算子
    prob = destroy_w / np.array(destroy_w).sum()
    seq = [i for i in range(len(destroy_w))]
    destroy_operator = np.random.choice(seq, size=1, p=prob)[0]

    # destroy操作
    removed_cities = destroy(destroy_operator, x, destroy_city_cnt)
    return x, removed_cities, destroy_operator

# 选择repair算子
def select_and_repair(repair_w, x, removed_cities):
    # 轮盘赌选择repair算子
    prob = repair_w / np.array(repair_w).sum()
    seq = [i for i in range(len(repair_w))]
    repair_operator = np.random.choice(seq, size=1, p=prob)[0]

    # repair操作: 此处设定repair_operator=1, 即只使用贪心插入
    repair(1, x, removed_cities)
    return x, repair_operator

# Adaptive Large Neighborhood Search (ALNS)
def calc_by_alns(dist_mat):
    # 模拟退火温度
    T = 100
    # 降温速度
    a = 0.97

    # destroy的城市数量
    destroy_city_cnt = int(len(dist_mat) * 0.1)
    # destroy算子的初始权重
    destroy_w = [1, 1, 1]
    # repair算子的初始权重
    repair_w = [1, 1]
    # destroy算子的使用次数
    destroy_cnt = [0, 0, 0]
    # repair算子的使用次数
    repair_cnt = [0, 0]
    # destroy算子的初始得分
    destroy_score = [1, 1, 1]
    # repair算子的初始得分
    repair_score = [1, 1]
    # destroy和repair的挥发系数
    lambda_rate = 0.5

    # 当前解,第一代, 贪心策略生成
    removed_cities = [i for i in range(dist_mat.shape[0])]
    x = []
    repair(1, x, removed_cities)

    # x = [i for i in range(dist_mat.shape[0])]

    # 历史最优解, 第一代和当前解相同, 注意是深拷贝, 此后有变化不影响x, 也不会因为x的变化而被影响
    history_best_x = copy.deepcopy(x)

    # 迭代
    cur_iter = 0
    max_iter = 1000
    print(f'cur_iter: {cur_iter}, best_dist: {dis_cal(history_best_x, dist_mat)}')
    while cur_iter < max_iter:
        # destroy算子
        destroyed_x, remove, destroy_operator_index = select_and_destroy(destroy_w, x, destroy_city_cnt)
        destroy_cnt[destroy_operator_index] += 1

        # repair算子
        test_x, repair_operator_index = select_and_repair(repair_w, destroyed_x, remove)
        repair_cnt[repair_operator_index] += 1

        if dis_cal(test_x, dist_mat) <= dis_cal(x, dist_mat):
            # 测试解更优, 更新当前解
            x = copy.deepcopy(test_x)
            if dis_cal(test_x, dist_mat) <= dis_cal(history_best_x, dist_mat):
                # 测试解为历史最优解, 更新历史最优解, 并设置最高的算子得分
                history_best_x = copy.deepcopy(test_x)
                destroy_score[destroy_operator_index] = 1.5
                repair_score[repair_operator_index] = 1.5
            else:
                # 测试解不是历史最优解, 但优于当前解, 设置第二高的算子得分
                destroy_score[destroy_operator_index] = 1.2
                repair_score[repair_operator_index] = 1.2
        else:
            if np.random.random() < np.exp((dis_cal(x, dist_mat) - dis_cal(test_x, dist_mat)) / T):
                # 当前解优于测试解, 但满足模拟退火条件, 更新当前解, 并设置第三高的算子得分
                x = copy.deepcopy(test_x)
                destroy_score[destroy_operator_index] = 0.8
                repair_score[repair_operator_index] = 0.8
            else:
                # 当前解不优于测试解, 不满足模拟退火条件, 保持当前解不变, 并设置最低的算子得分
                destroy_score[destroy_operator_index] = 0.5
                repair_score[repair_operator_index] = 0.5

        # 更新destroy算子的权重
        destroy_w[destroy_operator_index] = \
            destroy_w[destroy_operator_index] * lambda_rate + \
            (1 - lambda_rate) * destroy_score[destroy_operator_index] / destroy_cnt[destroy_operator_index]
        # 更新repair算子的权重
        repair_w[repair_operator_index] = \
            repair_w[repair_operator_index] * lambda_rate + \
            (1 - lambda_rate) * repair_score[repair_operator_index] / repair_cnt[repair_operator_index]
        # 降低温度
        T = a * T

        # 结束一轮迭代, 重置模拟退火初始温度
        cur_iter += 1
        print(f'cur_iter: {cur_iter}, best_dist: {dis_cal(history_best_x, dist_mat)}')

    # 打印ALNS得到的最优解
    print(f'ALNS得到的城市访问路径: \n{history_best_x}')
    print(f'ALNS得到的最优解距离: {dis_cal(history_best_x, dist_mat):.6f}')

# 读取tsp数据
def read_tsp_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        # 移除空行和注释行
        lines = [line.strip() for line in lines if line.strip()
                 and not line.startswith('NAME') and not line.startswith('COMMENT')
                 and not line.startswith('TYPE') and not line.startswith('DIMENSION')
                 and not line.startswith('EDGE_WEIGHT_TYPE')]
        # 提取坐标
        coordinates = []
        for line in lines:
            parts = line.split()
            if len(parts) == 3:
                index = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coordinates.append((x, y))
    return np.array(coordinates)


if __name__ == '__main__':
    # original_cities = pd.read_csv('data/cities.csv', header=0)
    # # 将经纬度转换为弧度
    # D = original_cities[['经度', '纬度']].values * math.pi / 180
    # cit_cnt = len(original_cities)
    # dist_mat = np.zeros((cit_cnt, cit_cnt))
    # for i in range(cit_cnt):
    #     for j in range(cit_cnt):
    #         if i == j:
    #             # 相同城市不允许访问
    #             dist_mat[i][j] = 1000000
    #         else:
    #             # 单位: km
    #             # 地球半径: 6378.14 km
    #             arg = math.cos(D[i][1]) * math.cos(D[j][1]) * math.cos(D[i][0] - D[j][0]) + \
    #                     math.sin(D[i][1]) * math.sin(D[j][1])
    #             dist_mat[i][j] = 6378.14 * math.acos(arg)
    # # ALNS求解TSP
    # start_time = time.time()
    # calc_by_alns(dist_mat)
    # end_time = time.time()
    # print(f'使用ALNS求解TSP, 耗时: {end_time - start_time:.2f} s')

    # 读取tsp数据
    original_cities = read_tsp_data('data/xqf131.tsp')
    dist_mat = np.zeros((len(original_cities), len(original_cities)))
    for i in range(len(original_cities)):
        for j in range(len(original_cities)):
            if i == j:
                # 相同城市不允许访问
                dist_mat[i][j] = 1000000
            else:
                dist_mat[i][j] = math.sqrt((original_cities[i][0] - original_cities[j][0]) ** 2 +
                                          (original_cities[i][1] - original_cities[j][1]) ** 2)
    # ALNS求解TSP
    start_time = time.time()
    calc_by_alns(dist_mat)
    end_time = time.time()
    print(f'使用ALNS求解TSP, 耗时: {end_time - start_time:.2f} s')


