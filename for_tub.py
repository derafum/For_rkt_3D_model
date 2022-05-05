
from mpl_toolkits.mplot3d import Axes3D
from math import *
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt2

# скорость источника
def speed_source(r):
    return 1/(4*pi*r*r)

# cos угла между вектором скорости и осью


# косинус между нормалью и осью


def unknown_z(a,c,x):
    # z = ((c**2)) - ((x**2)*(c**2)/(a**2))
    z = ((c)*(a - x**2))/(a)
    #print("a: ", a, "c: ", c, "x: ", x, "(sqrt(abs(z))): ", (sqrt(abs(z))  ))
    return (sqrt(abs(z)))

# коэффициенты
a = 100000000
b = 100000000
c = 100000000
n = 112


if n%2==0:
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
line = plt.plot(-x[0], y[0])
plt.setp(line, linestyle='-', color="g", linewidth=6)
line = plt.plot(x[0], y[0])
plt.setp(line, linestyle='-', color="g", linewidth=6)


# узнаем длину главной оси
len_line = round(2 * max(x[0]), 2)
# расстояние между точками
points_dist = len_line / (n - 1)
# начало отсчета
distance = -(max(x[0]))

print("Расстояние между точками на оси и длина линии:", points_dist, len_line)

# z координаты точек
z_coord_array = []

# рисуем точки
for i in range(n):
    if n==1:
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
        distance += points_dist
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
    coordinates_x.insert(i,0)
    z_coord_array.insert(i,z_coord )

# матрица с глaвными углами
cos_angles_array_main = np.zeros(n)

# матрица с cos углами для 1 матрицы
cos_angles_array = np.zeros(n)

# матрица с cos углами для 1 матрицы - cos (90-x)
cos_angles_array_2 = np.zeros(n)

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
        #cos_angles_array_main[z_coord] = -sqrt((1 - cos_angles ** 2))

        # считаем cos углов
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

        angle_v_n = cos(90 - (((np.arccos(cos_angles_array_main[j])) * 180 / np.pi) + (
                np.arccos(cos_angles_array_2[j]) * 180 / np.pi)))

        x_1 = speed_source(r) * angle_v_n

        v_ij_2 = x_1 * (cos_angles_array_main[j] + cos_angles_array_2[j])

        # v_new = speed_source(r) * angle_v_n

        speed_array[i, j] = x_1


print("Матрица с расстояниями между точек: ")
for i in distance_points_array:
    print(i)
print('=============================')

print("Матрица с cos углов (B): ", *cos_angles_array_main)


print("Матрица с V_i_j (A): ")
for i in speed_array:
    print(i)
print('=============================')

# матрица с Q
# Решение матричного уравнения (системы уравнений) Ax=B

res_del = np.linalg.solve(speed_array, cos_angles_array_main)

y_Array = [x for x in res_del]
x_Array = coordinates_x

print("Результат: ", res_del)
print("sum q: ", np.sum(res_del))
print('========================')
print('x: ', coordinates_x)

plt.title('3D модель')

fig = plt2.figure()
axes = fig.add_subplot(111)
for i in range(len(x_Array)):
    axes.scatter(x_Array[i], y_Array[i])

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
