"""采用双向A*的综合改进结果"""

import math
import numpy as np
import time
import matplotlib.pyplot as plt
from math import factorial

show_animation = True


class BidirectionalAStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.min_x, self.min_y = None, None
        self.max_x, self.max_y = None, None
        self.x_width, self.y_width, self.obstacle_map = None, None, None
        self.resolution = resolution
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:

        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        Bidirectional A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set_A, closed_set_A = dict(), dict()
        open_set_B, closed_set_B = dict(), dict()
        open_set_A[self.calc_grid_index(start_node)] = start_node
        open_set_B[self.calc_grid_index(goal_node)] = goal_node

        current_A = start_node
        current_B = goal_node
        meet_point_A, meet_point_B = None, None

        while 1:
            if len(open_set_A) == 0:
                print("Open set A is empty..")
                break

            if len(open_set_B) == 0:
                print("Open set B is empty..")
                break

            c_id_A = min(
                open_set_A,
                key=lambda o: self.find_total_cost(open_set_A, o, current_B))

            current_A = open_set_A[c_id_A]

            c_id_B = min(
                open_set_B,
                key=lambda o: self.find_total_cost(open_set_B, o, current_A))

            current_B = open_set_B[c_id_B]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current_A.x, self.min_x),
                         self.calc_grid_position(current_A.y, self.min_y),
                         "xc")
                plt.plot(self.calc_grid_position(current_B.x, self.min_x),
                         self.calc_grid_position(current_B.y, self.min_y),
                         "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set_A.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current_A.x == current_B.x and current_A.y == current_B.y:
                print("Found goal")
                meet_point_A = current_A
                meet_point_B = current_B
                break

            # Remove the item from the open set
            del open_set_A[c_id_A]
            del open_set_B[c_id_B]

            # Add it to the closed set
            closed_set_A[c_id_A] = current_A
            closed_set_B[c_id_B] = current_B

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):

                c_nodes = [
                    self.Node(current_A.x + self.motion[i][0],
                              current_A.y + self.motion[i][1],
                              current_A.cost + self.motion[i][2], c_id_A),
                    self.Node(current_B.x + self.motion[i][0],
                              current_B.y + self.motion[i][1],
                              current_B.cost + self.motion[i][2], c_id_B)
                ]

                n_ids = [
                    self.calc_grid_index(c_nodes[0]),
                    self.calc_grid_index(c_nodes[1])
                ]

                # If the node is not safe, do nothing
                continue_ = self.check_nodes_and_sets(c_nodes, closed_set_A,
                                                      closed_set_B, n_ids)

                if not continue_[0]:
                    if n_ids[0] not in open_set_A:
                        # discovered a new node
                        open_set_A[n_ids[0]] = c_nodes[0]
                    else:
                        if open_set_A[n_ids[0]].cost > c_nodes[0].cost:
                            # This path is the best until now. record it
                            open_set_A[n_ids[0]] = c_nodes[0]

                if not continue_[1]:
                    if n_ids[1] not in open_set_B:
                        # discovered a new node
                        open_set_B[n_ids[1]] = c_nodes[1]
                    else:
                        if open_set_B[n_ids[1]].cost > c_nodes[1].cost:
                            # This path is the best until now. record it
                            open_set_B[n_ids[1]] = c_nodes[1]

        rx, ry = self.calc_final_bidirectional_path(meet_point_A, meet_point_B,
                                                    closed_set_A, closed_set_B)

        return rx, ry

    # takes two sets and two meeting nodes and return the optimal path
    def calc_final_bidirectional_path(self, n1, n2, setA, setB):
        rx_A, ry_A = self.calc_final_path(n1, setA)
        rx_B, ry_B = self.calc_final_path(n2, setB)

        rx_A.reverse()
        ry_A.reverse()

        rx = rx_A + rx_B
        ry = ry_A + ry_B

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], \
                 [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def check_nodes_and_sets(self, c_nodes, closedSet_A, closedSet_B, n_ids):
        continue_ = [False, False]
        if not self.verify_node(c_nodes[0]) or n_ids[0] in closedSet_A:
            continue_[0] = True

        if not self.verify_node(c_nodes[1]) or n_ids[1] in closedSet_B:
            continue_[1] = True

        return continue_

    @staticmethod
    def calc_heuristic(n1, n2):
        h = math.hypot(n1.x - n2.x, n1.y - n2.y)
        if h >= 10:
            h *= 3.001
        else:
            h *= 0.801
        return h

    def find_total_cost(self, open_set, lambda_, n1):
        g_cost = open_set[lambda_].cost
        h_cost = self.calc_heuristic(n1, open_set[lambda_])
        f_cost = g_cost + h_cost
        return f_cost

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                  [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]

        return motion


##贝塞尔曲线运算函数
def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum(
        comb(n, i) * t**i * (1 - t)**(n - i) * points[i] for i in range(n + 1))


def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:, 0], new_points[:, 1]


##


def main():

    print(__file__ + '  start!')
    plt.title("Astar_improve")

    # start and goal position  [m]
    sx = -5.0
    sy = -5.0
    gx = 50
    gy = 50
    grid_size = 2.0
    robot_radius = 1.0

    # obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10)  # y坐标-10的一行-10~60的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 60):
        ox.append(i)
        oy.append(60)  # y坐标60的一行-10~60的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)  # x坐标-10的一列-10~61的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 61):
        ox.append(60)
        oy.append(i)  # x坐标60的一列-10~61的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)  # x坐标20的一列-10~40的坐标添加到列表并显示为黑色障碍物
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)  # x坐标40的一列20~60的坐标添加到列表并显示为黑色障碍物

    if show_animation:
        plt.plot(ox, oy, ".k")  # 黑色.       障碍物
        plt.plot(sx, sy, "og")  # 绿色圆圈    开始坐标
        plt.plot(gx, gy, "xb")  # 蓝色x       目标点
        plt.grid(True)
        plt.axis('equal')  # 保持栅格的横纵坐标刻度一致

    #路径计算，并计时

    a_star = BidirectionalAStarPlanner(
        ox, oy, grid_size, robot_radius)  # grid_size=resolution 初始化中传入的参数

    # 图形化显示
    start = time.perf_counter()
    pathx, pathy = a_star.planning(
        sx, sy, gx, gy)  # 开始与结束的坐标传入函数进行处理后，得到pathx,pathy：最终规划出的路径坐标
    end = time.perf_counter()
    print('用时' + str(1000 * (end - start)) + 'ms')

    #贝塞尔曲线显示
    # 贝塞尔曲线的控制点，为了方便更改，可根据出图效果调整
    points = np.array([[pathx[0], pathy[0]], [pathx[1], pathy[1]],
                       [pathx[2], pathy[2]], [pathx[3], pathy[3]],
                       [pathx[4], pathy[4]], [pathx[5], pathy[5]],
                       [pathx[6], pathy[6]], [pathx[7], pathy[7]],
                       [pathx[8], pathy[8]], [pathx[9], pathy[9]],
                       [pathx[10], pathy[10]], [pathx[11], pathy[11]],
                       [pathx[12], pathy[12]], [pathx[13], pathy[13]],
                       [pathx[14], pathy[14]], [pathx[15], pathy[15]]])
    points1 = np.array([[pathx[15], pathy[15]], [pathx[16], pathy[16]],
                        [pathx[17], pathy[17]]])
    points2 = np.array([[pathx[17], pathy[17]], [pathx[18], pathy[18]],
                        [pathx[19], pathy[19]], [pathx[20], pathy[20]],
                        [pathx[21], pathy[21]], [pathx[22], pathy[22]],
                        [pathx[23], pathy[23]], [pathx[24], pathy[24]],
                        [pathx[25], pathy[25]], [pathx[26], pathy[26]],
                        [pathx[27], pathy[27]]])
    points3 = np.array([[pathx[27], pathy[27]], [pathx[28], pathy[28]],
                        [pathx[29], pathy[29]]])
    points4 = np.array([[pathx[29], pathy[29]], [pathx[30], pathy[30]],
                        [pathx[31], pathy[31]], [pathx[32], pathy[32]],
                        [pathx[33], pathy[33]], [pathx[34], pathy[34]]])
    points5 = np.array([[pathx[35], pathy[35]], [pathx[36], pathy[36]],
                        [pathx[37], pathy[37]], [pathx[38], pathy[38]],
                        [pathx[39], pathy[39]], [pathx[40], pathy[40]],
                        [pathx[41], pathy[41]]])
    points6 = np.array([[pathx[41], pathy[41]], [pathx[42], pathy[42]],
                        [pathx[43], pathy[43]], [pathx[44], pathy[44]],
                        [pathx[45], pathy[45]], [pathx[46], pathy[46]],
                        [pathx[47], pathy[47]], [pathx[48], pathy[48]],
                        [pathx[49], pathy[49]], [pathx[50], pathy[50]],
                        [pathx[51], pathy[51]], [pathx[52], pathy[52]],
                        [pathx[53], pathy[53]], [pathx[54], pathy[54]],
                        [pathx[55], pathy[55]], [pathx[56], pathy[56]]])

    bx, by = evaluate_bezier(points, 50)
    bx1, by1 = evaluate_bezier(points1, 50)
    bx2, by2 = evaluate_bezier(points2, 50)
    bx3, by3 = evaluate_bezier(points3, 50)
    bx4, by4 = evaluate_bezier(points4, 50)
    bx5, by5 = evaluate_bezier(points5, 50)
    bx6, by6 = evaluate_bezier(points6, 50)

    if show_animation:
        plt.plot(pathx, pathy, "-r")  # 红色直线 最终路径
        plt.plot(bx, by, 'b-')  # 蓝色直线 贝塞尔曲线
        plt.plot(bx1, by1, 'b-')
        plt.plot(bx2, by2, 'b-')
        plt.plot(bx3, by3, 'b-')
        plt.plot(bx4, by4, 'b-')
        plt.plot(bx5, by5, 'b-')
        plt.plot(bx6, by6, 'b-')

        plt.show()
        plt.pause(0.001)  # 动态显示


# 
if __name__ == '__main__':
    main()