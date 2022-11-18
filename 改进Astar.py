"""
A* grid planning has not been improved
author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)
Modified by: Grizi-ju
Watch the video at bilibili, ID:小巨同学zz
Any questions, welcome the exchanges
See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import time
from math import factorial

show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):

        self.resolution = resolution  # grid resolution [m]
        self.rr = rr  # robot radius [m]
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    # 创建一个节点类，节点的信息包括：xy坐标，cost代价,parent_index
    class Node:

        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y
            self.cost = cost  # g(n)
            self.parent_index = parent_index  # index of previous Node

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    # 经过planning函数处理，传入sx,sy,gx,gy, 返回pathx,pathy(最终的路径)
    def planning(self, sx, sy, gx, gy):
        """
        1、sx = nstart  sy = ngoal
        2、open_set  closed_set
        3、open_set = nstart 
        4、将open表中代价最小的子节点=当前节点，并在plot上动态显示，按esc退出
        5、如果当前节点等于ngoal，提示找到目标点
        6、删除open表中的内容，添加到closed表中
        7、基于运动模型定义搜索方式
        8、pathx,pathy = 最终路径(传入ngoal,closed_set)，返回pathx,pathy
        """

        # 1、sx = nstart  sy = ngoal  初始化nstart、ngoal坐标作为一个节点，传入节点全部信息
        nstart = self.Node(
            self.calc_xyindex(sx, self.minx),  # position min_pos   2 (2.5)
            self.calc_xyindex(sy, self.miny),  # 2 (2.5)
            0.0,
            -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)
        # 2、open表、closed表设定为字典
        # 3、起点加入open表
        open_set, closed_set = dict(), dict()  # key - value: hash表
        open_set[self.calc_grid_index(nstart)] = nstart

        while 1:
            if len(open_set) == 0:
                print("Open_set is empty...")
                break

        # 4、将open表中代价最小的子节点 = 当前节点，并在plot上动态显示，按esc退出

        # f(n)=g(n)+h(n)  实际代价+预估代价
            c_id = min(open_set,
                       key=lambda o: open_set[o].cost + self.calc_heuristic(
                           ngoal, open_set[o]))
            current = open_set[c_id]

            # 将当前节点显示出来
            if show_animation:
                plt.plot(self.calc_grid_position(current.x, self.minx),
                         self.calc_grid_position(current.y, self.miny),
                         "xc")  # 青色x 搜索点
                # 按esc退出
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal!")
                ngoal.parent_index = current.parent_index
                print("ngoal_parent_index:", ngoal.parent_index)
                ngoal.cost = current.cost
                print("ngoal_cost:", ngoal.cost)
                break

            # 删除open表中的c_id的子节点,并把current添加到closed_set
            del open_set[c_id]
            closed_set[c_id] = current

            # 基于motion model做栅格扩展，也就是搜索方式，可进行改进，如使用双向搜索、JPS等
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(
                    current.x + move_x,  # 当前x+motion列表中第0个元素dx
                    current.y + move_y,
                    current.cost + move_cost,
                    c_id)
                n_id = self.calc_grid_index(node)  # 返回该节点位置index

                # 如果节点不可通过，跳过
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # 直接加入a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[
                            n_id] = node  # This path is the best until now. record it

        pathx, pathy = self.calc_final_path(ngoal, closed_set)

        return pathx, pathy

    def calc_final_path(self, ngoal,
                        closedset):  # 传入目标点和closed表，经过函数处理得到最终所有的xy列表
        pathx, pathy = [self.calc_grid_position(ngoal.x, self.minx)
                        ], [self.calc_grid_position(ngoal.y, self.miny)]
        parent_index = ngoal.parent_index
        while parent_index != -1:
            n = closedset[parent_index]
            pathx.append(self.calc_grid_position(n.x, self.minx))
            pathy.append(self.calc_grid_position(n.y, self.miny))
            parent_index = n.parent_index

        return pathx, pathy

    @staticmethod  # 静态方法，calc_heuristic函数不用传入self，因为要经常修改启发函数，目的是为了提高阅读性
    def calc_heuristic(n1, n2):  # n1: ngoal，n2: open_set[o]
        h = math.hypot(n1.x - n2.x, n1.y - n2.y)  #欧几里得距离
        #h =  max(n1.x-n2.x,n1.y-n2.y)#切比雪夫距离

        if h > 18:
            h *= 3.001
 
        return h

    # 得到全局地图中的具体坐标: 传入地图中最小处障碍物的pos和index
    def calc_grid_position(self, index, minpos):
        pos = index * self.resolution + minpos
        return pos

    # 位置转化为以栅格大小为单位的索引: 传入position,min_pos
    def calc_xyindex(self, position, min_pos):
        return round(
            (position - min_pos) /
            self.resolution)  # (当前节点-最小处的坐标)/分辨率=pos_index  round四舍五入向下取整

    # 计算栅格地图节点的index： 传入某个节点
    def calc_grid_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    # 验证是否为可通行节点
    def verify_node(self, node):
        posx = self.calc_grid_position(node.x, self.minx)
        posy = self.calc_grid_position(node.y, self.miny)

        if posx < self.minx:
            return False
        elif posy < self.miny:
            return False
        elif posx >= self.maxx:
            return False
        elif posy >= self.maxy:
            return False

        if self.obmap[int(node.x)][int(node.y)]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        self.minx = round(min(ox))  # 地图中的临界值 -10
        self.miny = round(min(oy))  # -10
        self.maxx = round(max(ox))  # 60
        self.maxy = round(max(oy))  # 60
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx) / self.resolution)  # 35
        self.ywidth = round((self.maxy - self.miny) / self.resolution)  # 35
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        self.obmap = [[False for i in range(int(self.ywidth))]
                      for i in range(int(self.xwidth))]
        for ix in range(int(self.xwidth)):
            x = self.calc_grid_position(ix, self.minx)
            for iy in range(int(self.ywidth)):
                y = self.calc_grid_position(iy, self.miny)
                for iox, ioy in zip(ox, oy):  #将ox,oy打包成元组，返回列表，并遍历
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:  #代价小于车辆半径，可正常通过，不会穿越障碍物
                        self.obmap[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [#舍弃三个方向
                    [1, 0, 1],
                    [0, 1, 1],
                    #[-1, 0, 1],
                    [0, -1, 1],
                    [1, 1, math.sqrt(2)],
                    [1, -1, math.sqrt(2)],
                    #[-1, 1,math.sqrt(2)],
                    #[-1, -1, math.sqrt(2)]
                 ]
        # motion = [  #邻域扩展法<32邻域>
        #     [1, 0, 1],  #0
        #     [0, 1, 1],
        #     [-1, 0, 1],
        #     [0, -1, 1],
        #     [1, 1, math.sqrt(2)],
        #     [1, -1, math.sqrt(2)],
        #     [-1, 1, math.sqrt(2)],
        #     [-1, -1, math.sqrt(2)],
        #     [-2, -3, 2*math.sqrt(2)+1],  #1
        #     [-2, 3,  2*math.sqrt(2)+1],
        #     [2, -3,  2*math.sqrt(2)+1],
        #     [2, 3,  2*math.sqrt(2)+1],
        #     [-3, 2,  2*math.sqrt(2)+1],
        #     [-3, -2,  2*math.sqrt(2)+1],
        #     [3, -2,  2*math.sqrt(2)+1],
        #     [3, 2, 2*math.sqrt(2)+1],
        #     [-1, -3, math.sqrt(2)+2],  #2 
        #     [-1, 3, math.sqrt(2)+2],
        #     [1, -3, math.sqrt(2)+2],
        #     [1, 3, math.sqrt(2)+2],
        #     [-3, 1, math.sqrt(2)+2],
        #     [-3, -1, math.sqrt(2)+2],
        #     [3, 1, math.sqrt(2)+2],
        #     [3, -1, math.sqrt(2)+2],
        #     [-1, -2, math.sqrt(2)+1],  #3
        #     [-1, 2, math.sqrt(2)+1],
        #     [1, -2, math.sqrt(2)+1],
        #     [1, 2, math.sqrt(2)+1],
        #     [2, 1, math.sqrt(2)+1],
        #     [2, -1, math.sqrt(2)+1],
        #     [-2, 1, math.sqrt(2)+1],
        #     [-2, -1, math.sqrt(2)+1]
        # ]
        return motion
##贝塞尔曲线运算函数
def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum(comb(n, i)*t**i * (1-t)**(n-i)*points[i] for i in range(n+1))

def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:, 0], new_points[:, 1]
##

def main():

    print(__file__ + '  start!')
    plt.title("Weighing Coeffient + Neighborhood optimization +  Bezier Curves")

    # start and goal position  [m]
    sx = -6.0
    sy = -6.0
    gx = 50
    gy = 50
    grid_size = 2.0
    robot_radius = 1.0

    # obstacle positions
    ox, oy = [],[]
    for i in range(-10, 60): 
        ox.append(i)
        oy.append(-10)      # y坐标-10的一行-10~60的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 60):
        ox.append(i)
        oy.append(60)       # y坐标60的一行-10~60的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)        # x坐标-10的一列-10~61的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 61):
        ox.append(60)
        oy.append(i)        # x坐标60的一列-10~61的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)        # x坐标20的一列-10~40的坐标添加到列表并显示为黑色障碍物
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)   # x坐标40的一列20~60的坐标添加到列表并显示为黑色障碍物



    if show_animation:
        plt.plot(ox, oy, ".k")  # 黑色.       障碍物
        plt.plot(sx, sy, "og")  # 绿色圆圈    开始坐标
        plt.plot(gx, gy, "xb")  # 蓝色x       目标点
        plt.grid(True)
        plt.axis('equal')  # 保持栅格的横纵坐标刻度一致

    #路径计算，并计时
    
    a_star = AStarPlanner(ox, oy, grid_size,
                          robot_radius)  # grid_size=resolution 初始化中传入的参数

    # 图形化显示
    start = time.perf_counter()
    pathx, pathy = a_star.planning(
        sx, sy, gx, gy)  # 开始与结束的坐标传入函数进行处理后，得到pathx,pathy：最终规划出的路径坐标
    end = time.perf_counter()
    print('用时' + str(1000 * (end - start)) + 'ms')

    #贝塞尔曲线显示
     # 贝塞尔曲线的控制点，为了方便更改，可根据出图效果调整
    points = np.array([[pathx[0], pathy[0]], [pathx[1], pathy[1]], [pathx[2], pathy[2]],
                       [pathx[3], pathy[3]], [pathx[4], pathy[4]], [pathx[5], pathy[5]],
                       [pathx[6], pathy[6]], [pathx[7], pathy[7]], [pathx[8], pathy[8]],
                       [pathx[9], pathy[9]], [pathx[10], pathy[10]], [pathx[11], pathy[11]],
                       [pathx[12], pathy[12]], [pathx[13], pathy[13]], [pathx[14], pathy[14]],
                       [pathx[15], pathy[15]]])
    points1 = np.array([[pathx[15], pathy[15]], [pathx[16], pathy[16]], [pathx[17], pathy[17]]])
    points2 = np.array([[pathx[17], pathy[17]], [pathx[18], pathy[18]], [pathx[19], pathy[19]],
                        [pathx[20], pathy[20]], [pathx[21], pathy[21]], [pathx[22], pathy[22]],
                        [pathx[23], pathy[23]], [pathx[24], pathy[24]], [pathx[25], pathy[25]],
                        [pathx[26], pathy[26]], [pathx[27], pathy[27]]])
    points3 = np.array([[pathx[27], pathy[27]], [pathx[28], pathy[28]], [pathx[29], pathy[29]]])
    points4 = np.array([[pathx[29], pathy[29]], [pathx[30], pathy[30]], [pathx[31], pathy[31]],
                        [pathx[32], pathy[32]], [pathx[33], pathy[33]], [pathx[34], pathy[34]],
                        [pathx[35], pathy[35]], [pathx[36], pathy[36]], [pathx[37], pathy[37]],
                        [pathx[38], pathy[38]], [pathx[39], pathy[39]], [pathx[40], pathy[40]],
                        [pathx[41], pathy[41]], [pathx[42], pathy[42]], [pathx[43], pathy[43]],
                        [pathx[44], pathy[44]], [pathx[45], pathy[45]], [pathx[46], pathy[46]],
                        [pathx[47], pathy[47]], [pathx[48], pathy[48]], [pathx[49], pathy[49]],
                        [pathx[50], pathy[50]], [pathx[51], pathy[51]], [pathx[52], pathy[52]]])

    bx, by = evaluate_bezier(points, 50)
    bx1, by1 = evaluate_bezier(points1, 50)
    bx2, by2 = evaluate_bezier(points2, 50)
    bx3, by3 = evaluate_bezier(points3, 50)
    bx4, by4 = evaluate_bezier(points4, 50)

    if show_animation:
        plt.plot(pathx, pathy, "-r")  # 红色直线 最终路径
        plt.plot(bx, by, 'b-')  # 蓝色直线 贝塞尔曲线
        plt.plot(bx1, by1, 'b-')
        plt.plot(bx2, by2, 'b-')
        plt.plot(bx3, by3, 'b-')
        plt.plot(bx4, by4, 'b-')

        plt.show()
        plt.pause(0.001)  # 动态显示


    #原图形化显示
    # if show_animation:
    #     plt.plot(pathx, pathy, "-r")  # 红色直线 最终路径
    #     plt.show()
    #     plt.pause(0.001)  # 动态显示


if __name__ == '__main__':
    main()