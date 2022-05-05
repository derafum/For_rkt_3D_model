# pyinstaller -F -w  main.py

import os
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from mpl_toolkits.mplot3d import Axes3D
from math import *
import matplotlib.pyplot as plt
import numpy as np

from numpy import linalg as LA
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


global line1, line2

def summation():
    global x_Array, y_Array, vr_array_y, cp_array_y, res_del, speed_array, cp_array_y, vr_array_y, line1, line2,a,b,c,n, V_t_i

    a = int(num1.get())
    b = int(num2.get())
    c = int(num3.get())
    n = int(num3.get())

    if n % 2 == 0:
        f = True
    else:
        f = False

    '''a = (a)**2 * 0.5
    b = (b)**2 * 0.5
    c = (c)**2 * 0.5'''

    #fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    #ax = fig.add_subplot(111, projection='3d')

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
    #ax.plot_wireframe(x, y, z, rstride=18, cstride=18, color='b')

    # рисуем точку 0 0 0
    #ax.scatter([0], [0], [0], color="black", s=50)

    # координаты точек X
    coordinates_x = []

    # рисуем 2 линии осей
    #line1 = plt.plot(-x[0], y[0])
   # plt.setp(line1, linestyle='-', color="g", linewidth=6)
    #line2 = plt.plot(x[0], y[0])
  #  plt.setp(line2, linestyle='-', color="g", linewidth=6)

    # узнаем длину главной оси
    len_line = round(2 * max(x[0]), 2)
    # расстояние между точками
    points_dist = len_line / (n - 1)


    # кол-во секторов * 2

    sectaries = n * 2
    len_sectaries = len_line/sectaries

    # начало отсчета
    distance = -(max(x[0])) + len_sectaries


    print("Расстояние между точками на оси и длина линии:", points_dist, len_line)

    # z координаты точек
    z_coord_array = []

    # рисуем точки
    for i in range(n):
        if n == 1:
           # ax.scatter([0], [0], [0], color="r", s=50)
            z_coord = unknown_z(a, c, 0)
           # ax.scatter(0, [0], z_coord, color="g", s=50)
            #ax.scatter(0, [0], -z_coord, color="g", s=50)
            z_coord = unknown_z(a, c, 0)
            z_coord_array.append(z_coord)
            coordinates_x.append(0)
        else:
            # рисуем точки на оси OX
           # ax.scatter(distance, [0], [0], color="r", s=50)
            # считаем координату z
            z_coord = unknown_z(a, c, distance)
            # условие чтобы не рисовать крайние точки
            #if (i != 0 or i != n - 1):
             #    ax.scatter(distance, [0], z_coord, color="g", s=50)
             #   ax.scatter(distance, [0], -z_coord, color="g", s=50)
            if i < int(n / 2):
                z_coord_array.append(z_coord)
                coordinates_x.append(distance)
            distance += len_sectaries*2
    t = 0
    for i in range(len(coordinates_x), 0, -1):
        coordinates_x.append(-coordinates_x[i - 1])
        t += 1
    z_coord_array += z_coord_array[::-1]

    # вставляем в середину массивов элемент

    if not f:
        len_arr = len(z_coord_array)
        i = len_arr // 2
        z_coord = unknown_z(a, c, 0)
        coordinates_x.insert(i, 0)
        z_coord_array.insert(i, z_coord)

    # матрица с глaвными углами
    cos_angles_array_main = np.zeros(n)

    # матрица с cos углами для 1 матрицы
    cos_angles_array = np.zeros(n)

    # матрица с cos углами для 1 матрицы - cos (90-x)
    cos_angles_array_2 = np.zeros(n)

    # матрица с cos углами гамма
    cos_angles_array_gamma = np.zeros(n)
    # 2 блок
    # считаем углы
    for z_coord in range(int(len(z_coord_array) / 2)):

        cathet = z_coord_array[z_coord + 1] - z_coord_array[z_coord]
        cathet_2 = z_coord_array[z_coord + 1]

        # считаем cos углов для 2 этапа
        if n == 1:
            cos_angles_array_2[z_coord] = 0
            cos_angles_array_main[z_coord] = 0
            cos_angles_array[z_coord] = 0
        else:
            hypotenuse = sqrt(cathet ** 2 + points_dist ** 2)
            cos_angles = points_dist / hypotenuse
            cos_angles_array_main[z_coord] = cos_angles

            # считаем cos углов
            hypotenuse_2 = sqrt(cathet_2 ** 2 + points_dist ** 2)
            cos_angles_array[z_coord] = points_dist / hypotenuse_2

            cos_angles_array_2[z_coord] = -sqrt(1 - cos_angles ** 2)
            #cos_angles_array_2[z_coord] = sqrt(1 - cos_angles ** 2)
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
            if n == 1:
                r = sqrt(((z_coord_array[0]) ** 2))
            else:
                r = sqrt(((j - i) * points_dist) ** 2 + ((z_coord_array[j]) ** 2))
            distance_points_array[i, j] = r
            # print("true:", 'i: ', i, 'j:', j)

            #
            angle = cos_angles_array[i]
            v_ix = speed_source(r) * angle
            v_jy = speed_source(r) * sqrt((1 - angle ** 2))
            v_ij = v_ix * cos_angles_array_2[j] + v_jy * cos_angles_array_main[j]

            v_ij = v_ix * cos_angles_array_main[i] + v_jy * cos_angles_array_2[j]


            if i==j:
                angle_v_n = cos_angles_array_main[j]
            elif i>j:
                angle_v_n = cos(90 - (((np.arccos(cos_angles_array_main[j])) * 180 / np.pi) + (
                        np.arccos(cos_angles_array_2[j]) * 180 / np.pi)))
            else:
                angle_v_n = cos(90 - ((((np.arccos(cos_angles_array_main[j])) * 180 / np.pi) + (
                        np.arccos(cos_angles_array_2[j]) * 180 / np.pi))))
            x_1 = speed_source(r) * angle_v_n



            v_ij_2 = x_1 * (cos_angles_array_main[j] + cos_angles_array_2[j])

            # v_new = speed_source(r) * angle_v_n

            speed_array[i, j] = x_1

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
    # Решение матричного уравнения (системы уравнений) Ax=B

    res_del = np.linalg.solve(speed_array, cos_angles_array_main)
    y_Array=[]
    #cr_len = round(n/2)-1
    cr_len = int (n/2)
    y_Array = [x  for x in res_del]

    for y in y_Array:
        if y == y_Array[cr_len]:
            y_Array[cr_len] = max(y_Array)
    x_Array = coordinates_x

    vr_array_y = []
    for i in range(len(y_Array)):
        vr = y_Array[i] / (2 * 3.14 * (sum(z_coord_array)))
        vr_array_y.append(vr)


    for y in vr_array_y:
        if y == vr_array_y[cr_len]:
            vr_array_y[cr_len] = vr_array_y[cr_len-1]

    '''
    cp_array_y = []
    
    for i in range(n):
        V_t_i = sqrt(1 - cos_angles_array_main[i] ** 2) + vr_array_y[i] * cos_angles_array_main[i] + (
                    (sum(vr_array_y) - vr_array_y[i]) * cos(
                90 - (((np.arccos(cos_angles_array_main[i])) * 180 / np.pi) + (
                        np.arccos(cos_angles_array_2[i]) * 180 / np.pi))))
        cp = 1 -  (V_t_i**2)
        cp_array_y.append(cp)

    V_t_i = 0
    cp_test = 0'''
    ''' 
    for i in range(len(y_Array)):
        for j in range(len(y_Array)):
            test = 3
            V_t_i += ( ((i==j)* vr_array_y[j]* sqrt(1 - cos_angles_array_main[j]**2))  + ( ((i<j)*vr_array_y[j]) * ((np.arccos(cos_angles_array_main[j])) * 180 / np.pi) - (
                        np.arccos(cos_angles_array_2[j]) * 180 / np.pi) ) - ((i>j) * vr_array_y[j] * ((((np.arccos(cos_angles_array_main[j])) * 180 / np.pi) + (
                        np.arccos(cos_angles_array_2[j]) * 180 / np.pi))) ) + cos_angles_array_main[j]*1)
            cp_test += vr_array_y[j] * sqrt(1 - cos_angles_array_2[j]**2)
        cp = 1 - (V_t_i ** 2)
        cp = 1 - ( cp_test +  cos_angles_array_main[i] ) ** 2
        cp_array_y.append(cp)'''

    cp_array_y = []


    for i in range(len(y_Array)):
        #cp = 1 -  (vr_array_y[i]**2)
        cp = 1 - ((9/4)*(sqrt(1-cos_angles_array_main[i]))**2)
        cp_array_y.append(cp)



    print("Результат (Q): ", res_del)
    print('========================')
    print('x: ', coordinates_x)
    print('z_coord_array: ', z_coord_array)



    print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\t")
    print('test')
    for i in range(n):
        test = -sqrt(1 - cos_angles_array_main[i]**2) + vr_array_y[i] * cos_angles_array_main[i] + ( (sum(vr_array_y) - vr_array_y[i] )  * cos(90 - (((np.arccos(cos_angles_array_main[i])) * 180 / np.pi) + (
                        np.arccos(cos_angles_array_2[i]) * 180 / np.pi))) )
        print(f"{i}", test)






    # ax.scatter(0.0764, [0], 0.099, color="r", s=100)
    # ax.scatter(-max(x[0]), [0], [0], color="b", s=200)

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    # print('max_radius: ',len(x))
    # print('max(x[0]: ',max(x[0]))
    # print('min(x[0]: ',min(x[0]))
    # print("sum: ",len_line,points_dist)


def clicked_save_1():
    with open('result_Q.txt', 'w') as file:
        anon = lambda arr: '\n'.join(' '.join(map(str, line)) for line in arr)
        file.write(f"Результаты (Q): {' '.join(map(str, res_del)) }\n\n\n")
        file.write(f"Результаты (V_i_j): {anon(speed_array)}\n\n")
    os.system('start result_Q.txt')

def clicked_save_2():

    with open('result_Vr.txt', 'w') as file:
        anon = lambda arr: '\n'.join(' '.join(map(str, line)) for line in arr)
        file.write(f"Результаты (Vr): {' '.join(map(str, vr_array_y)) }\n\n\n")
        file.write(f"Результаты (V_i_j): {anon(speed_array)}\n\n")
    os.system('start result_Vr.txt')

def clicked_save_3():

    with open('result_Cp.txt', 'w') as file:
        anon = lambda arr: '\n'.join(' '.join(map(str, line)) for line in arr)
        file.write(f"Результаты (Cp): {' '.join(map(str, cp_array_y)) }\n\n\n")
        file.write(f"Результаты (V_i_j): {anon(speed_array)}\n\n")
    os.system('start result_Cp.txt')

win = tk.Tk()
win.title("Исследование обтекания осесимметричного тела")
win.geometry("400x400")



res = tk.Label(win, text = "Введите а:")
res.grid(row = 0, column = 1)

num1 = tk.Entry(win)
num1.grid(row = 0, column = 2)


res = tk.Label(win, text = "Введите b:")
res.grid(row = 2, column = 1)

num2 = tk.Entry(win)
num2.grid(row = 2, column = 2)

res = tk.Label(win, text = "Введите c:")
res.grid(row = 4, column = 1)


num4 = tk.Entry(win)
num4.grid(row = 4, column = 2)


res = tk.Label(win, text = "Введите n:")
res.grid(row = 5, column = 1)

num3 = tk.Entry(win)
num3.grid(row = 5, column = 2)




def clicked1():
    global x_Array, y_Array
    fig = plt2.figure()
    axes = fig.add_subplot(111)
    for i in range(len(x_Array)):
        axes.scatter(x_Array[i], y_Array[i])
    # y_Array = [abs(x) for x in y_Array]
    axes.plot(x_Array, y_Array)
    plt2.xlabel('Значениe X:', fontsize=12)
    plt2.ylabel('значение Q', fontsize=12)
    plt2.title('График')
    plt2.show()


def clicked2():
    global x_Array, vr_array_y
    fig = plt2.figure()
    axes2 = fig.add_subplot(111)
    for i in range(len(x_Array)):
        axes2.scatter(x_Array[i], vr_array_y[i])
    axes2.plot(x_Array, vr_array_y)
    plt2.xlabel('Значениe X:', fontsize=12)
    plt2.ylabel('значение Vr', fontsize=12)
    plt2.title('График')
    plt2.show()


def clicked3():
    global x_Array, cp_array_y
    fig = plt2.figure()
    axes3 = fig.add_subplot(111)
    for i in range(len(x_Array)):
        axes3.scatter(x_Array[i], cp_array_y[i])
    # y_Array = [abs(x) for x in y_Array]
    axes3.plot(x_Array, cp_array_y)
    plt2.xlabel('Значениe X:', fontsize=12)
    plt2.ylabel('значение Cp', fontsize=12)
    plt2.title('График')
    plt2.show()


def view_3d():
    if n % 2 == 0:
        f = True
    else:
        f = False

    '''a = (a)**2 * 0.5
    b = (b)**2 * 0.5
    c = (c)**2 * 0.5'''

    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')

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

    # рисуем точку 0 0 0
    ax.scatter([0], [0], [0], color="black", s=50)

    # координаты точек X
    coordinates_x = []

    # рисуем 2 линии осей
    line1 = plt.plot(-x[0], y[0])
    plt.setp(line1, linestyle='-', color="g", linewidth=6)
    line2 = plt.plot(x[0], y[0])
    plt.setp(line2, linestyle='-', color="g", linewidth=6)

    # z координаты точек
    z_coord_array = []
    # кол-во секторов * 2

    # узнаем длину главной оси
    len_line = round(2 * max(x[0]), 2)
    # расстояние между точками
    points_dist = len_line / (n - 1)

    sectaries = n * 2
    len_sectaries = len_line / sectaries

    # начало отсчета
    distance = -(max(x[0])) + len_sectaries

    # рисуем точки
    for i in range(n):
        if n == 1:
            ax.scatter([0], [0], [0], color="r", s=50)
            z_coord = unknown_z(a, c, 0)
            ax.scatter(0, [0], z_coord, color="g", s=50)
            ax.scatter(0, [0], -z_coord, color="g", s=50)
            z_coord = unknown_z(a, c, 0)
            z_coord_array.append(z_coord)
            coordinates_x.append(0)
        else:
            # рисуем точки на оси OX
            ax.scatter(distance, [0], [0], color="r", s=50)
            # считаем координату z
            z_coord = unknown_z(a, c, distance)
            # условие чтобы не рисовать крайние точки
            if (i != 0 or i != n - 1):
                ax.scatter(distance, [0], z_coord, color="g", s=50)
                ax.scatter(distance, [0], -z_coord, color="g", s=50)
            if i < int(n / 2):
                z_coord_array.append(z_coord)
                coordinates_x.append(distance)
            distance += len_sectaries * 2

    plt.title('3D модель')

    plt.show()

rad1 = Button(win, text='График Q(x): ', command=clicked1)
rad1.grid(row=10, column=1)


rad2 = Button(win, text='График Vr(x): ', command=clicked2)
rad2.grid(row=11, column=1)


rad3 = Button(win, text='График Cp(x): ', command=clicked3)
rad3.grid(row=12, column=1)





btn = Button(win, text="Посмотреть результаты (Q):", command = clicked_save_1)
btn.grid(row= 10, column=2)

btn = Button(win, text="Посмотреть результаты (Vr):", command = clicked_save_2)
btn.grid(row= 11, column=2)

btn = Button(win, text="Посмотреть результаты (Cp):", command = clicked_save_3)
btn.grid(row= 12, column=2)

button = Button(win, text = "Рассчитать: ", command = summation)
button.grid(row = 13, column = 2)

button = Button(win, text = "Показать 3D модель: ", command = view_3d)
button.grid(row = 2, column = 5)


win.mainloop()