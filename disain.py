import os
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, pi
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt2

# скорость источника
def speed_source(r):
    return 1/(4*pi*r*r)

# cos угла между вектором скорости и осью
def cos_vectors(v,x_y):
    # что-то делаем
    return # значение cos угла

# косинус между нормалью и осью
def cos_vectors_n(n,x):
    # что-то делаем
    return # значение cos угла

def unknown_z(a,c,x):
    # z = ((c**2)) - ((x**2)*(c**2)/(a**2))
    z = ((c)*(a - x**2))/(a)
    #print("a: ", a, "c: ", c, "x: ", x, "(sqrt(abs(z))): ", (sqrt(abs(z))  ))
    return (sqrt(abs(z)))




def summation():
    global res_del,speed_array
    a = int(num1.get())
    b = int(num2.get())
    n = int(num3.get())

    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')

    c = 100

    coefs = (a, b, c)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
    # Radii corresponding to the coefficients:
    rx, ry, rz = np.sqrt(coefs)  #

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_wireframe(x, y, z, rstride=18, cstride=18, color='b')

    ax.scatter([0], [0], [0], color="black", s=50)

    coordinates_x = []

    # рисуем 2 линии
    line = plt.plot(-x[0], y[0])
    plt.setp(line, linestyle='-', color="g", linewidth=6)
    line = plt.plot(x[0], y[0])
    plt.setp(line, linestyle='-', color="g", linewidth=6)

    len_line = round(2 * max(x[0]), 2)
    points_dist = len_line / (n - 1)  # расстояние между точками
    distance = -(max(x[0]))

    print("Расстояние между точками на оси:", points_dist, len_line, distance)

    z_coord_array = []

    # рисуем точки
    for i in range(n):
        ax.scatter(distance, [0], [0], color="r", s=50)
        z_coord = unknown_z(a, c, distance)
        if (round(z_coord) != 0):
            ax.scatter(distance, [0], z_coord, color="g", s=50)
            ax.scatter(distance, [0], -z_coord, color="g", s=50)
        if i < int(n / 2):
            z_coord_array.append(z_coord)
            coordinates_x.append(distance)
        # print(f'distance {i}',distance)
        # print(f'x {i}', )
        distance += points_dist
    t = 0
    for i in range(len(coordinates_x), 0, -1):
        coordinates_x.append(-coordinates_x[i - 1])
        t += 1
    z_coord_array += z_coord_array[::-1]

    # матрица с глaвными углами
    cos_angles_array_main = np.zeros(n)

    # матрица с cos углами для 1 матрицы
    cos_angles_array = np.zeros(n)

    # матрица с cos углами для 1 матрицы - cos (90-x)
    cos_angles_array_2 = np.zeros(n)

    for z_coord in range(int(len(z_coord_array) / 2)):
        try:
            cathet = z_coord_array[z_coord + 1] - z_coord_array[z_coord]
            cathet_2 = z_coord_array[z_coord + 1]
        except:
            pass
        hypotenuse = sqrt(cathet ** 2 + points_dist ** 2)
        cos_angles = points_dist / hypotenuse
        cos_angles_array_main[z_coord] = cos_angles

        hypotenuse_2 = sqrt(cathet_2 ** 2 + points_dist ** 2)
        cos_angles_array[z_coord] = points_dist / hypotenuse_2

        cos_angles_array_2[z_coord] = sqrt(1 - cos_angles ** 2)
        # print("cos_angles: ", cos_angles, "distance: ", distance, "hypotenuse: ", hypotenuse, "z_coord: ", z_coord)

    cos_angles_array_main += cos_angles_array_main[::-1]

    cos_angles_array += cos_angles_array[::-1]

    cos_angles_array_2 += cos_angles_array_2[::-1]

    # матрица с R
    distance_points_array = np.zeros((n, n))
    # матрица с V(ij)
    speed_array = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            r = sqrt(((i - j) * points_dist) ** 2 + ((z_coord_array[j]) ** 2))
            distance_points_array[i, j] = r
            # print("true:", 'i: ', i, 'j:', j)

            #
            angle = cos_angles_array[j]
            v_ix = speed_source(r) * angle
            v_jy = speed_source(r) * sqrt((1 - angle ** 2))

            v_ij = v_ix * cos_angles_array_main[j] + v_jy * cos_angles_array_2[j]

            speed_array[i, j] = v_ij

    print("Матрица с cos углов: ", *cos_angles_array_main)
    print("Матрица с расстояниями между точек: ")
    for i in distance_points_array:
        print(i)
    print('=============================')

    print("Матрица с V_i_j: ")
    for i in speed_array:
        print(i)
    print('=============================')

    # матрица с Q
    res_del = np.linalg.solve(speed_array, cos_angles_array_main)

    y_Array = [x for x in res_del]
    x_Array = coordinates_x

    print("Результат: ", res_del)
    print('========================')
    print('x: ', coordinates_x)

    plt.title('3D модель')

    fig = plt2.figure()
    axes = fig.add_subplot(111)
    axes.plot(x_Array, y_Array)
    plt2.xlabel('Значениe X:', fontsize=12)
    plt2.ylabel('значение Q', fontsize=12)
    plt2.title('График')
    plt2.show()
    plt.show()

    # ax.scatter(0.0764, [0], 0.099, color="r", s=100)
    # ax.scatter(-max(x[0]), [0], [0], color="b", s=200)

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
    # print('max_radius: ',len(x))
    # print('max(x[0]: ',max(x[0]))
    # print('min(x[0]: ',min(x[0]))
    # print("sum: ",len_line,points_dist)


def clicked():

    with open('result.txt', 'w') as file:
        anon = lambda arr: '\n'.join(' '.join(map(str, line)) for line in arr)
        file.write(f"Результаты (Q): {' '.join(map(str, res_del)) }\n\n\n")
        file.write(f"Результаты (V_i_j): {anon(speed_array)}\n\n")
    os.system('start result.txt')

win = tk.Tk()
win.title("Исследование обтекания осесимметричного тела")
win.geometry("400x400")



res = tk.Label(win, text = "Введите а:")
res.grid(row = 0, column = 2)

num1 = tk.Entry(win)
num1.grid(row = 0, column = 3)


res = tk.Label(win, text = "Введите b:")
res.grid(row = 2, column = 2)

num2 = tk.Entry(win)
num2.grid(row = 2, column = 3)

res = tk.Label(win, text = "Введите n:")
res.grid(row = 3, column = 2)

num3 = tk.Entry(win)
num3.grid(row = 3, column = 3)



btn = Button(win, text="Посмотреть результаты:", command=clicked)
btn.grid(column=2, row=6)


button = tk.Button(win, text = "Рассчитать", command = summation)
button.grid(row = 4, column = 3)
win.mainloop()