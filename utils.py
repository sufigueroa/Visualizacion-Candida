from numpy import cos, sin
import numpy as np
import math
from PIL import Image
import cv2
from skspatial.objects import Line, Plane, Vector
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

from params import *


###################################################
### Funciones Generales
###################################################

# https://stackoverflow.com/questions/18602525/python-pil-for-loop-to-work-with-multi-image-tiff
# Funcion para abrir la imagen tiff y dejarla como una matriz
def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

def save_image(img, path):
    img = Image.fromarray(np.uint8(img), 'L')
    img.save(path)

def save_video(frames, path):
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frames[0].shape[1], frames[0].shape[0]), 0)
    for frame in frames:
        video.write(np.uint8(frame))
    video.release()

###################################################


###################################################
### Funciones de Interpolacion
###################################################

def isotropic_interpolation(matrix):
    new = []
    for i in range(len(matrix) - 1):
        img1 = matrix[i]
        img2 = matrix[i + 1]
        ab = np.dstack((img1, img2))
        inter = np.mean(ab, axis=2, dtype=ab.dtype) 
        new.append(img1)
        new.append(inter)
    new.append(img2)
    return np.array(new)

###################################################


###################################################
### Parte A:
###################################################

#https://neurostars.org/t/trilinear-interpolation-in-python/18019
def trilinear(xyz, data):
    '''
    xyz: array with coordinates inside data
    data: 3d volume
    returns: interpolated data values at coordinates
    '''
    ijk = xyz.astype(np.int32)
    i, j, k = ijk[:,0], ijk[:,1], ijk[:,2]
    V000 = data[ i   , j   ,  k   ].astype(np.int32)
    V100 = data[(i+1), j   ,  k   ].astype(np.int32)
    V010 = data[ i   ,(j+1),  k   ].astype(np.int32)
    V001 = data[ i   , j   , (k+1)].astype(np.int32)
    V101 = data[(i+1), j   , (k+1)].astype(np.int32)
    V011 = data[ i   ,(j+1), (k+1)].astype(np.int32)
    V110 = data[(i+1),(j+1),  k   ].astype(np.int32)
    V111 = data[(i+1),(j+1), (k+1)].astype(np.int32)
    xyz = xyz - ijk
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    Vxyz = (V000 * (1 - x)*(1 - y)*(1 - z)
            + V100 * x * (1 - y) * (1 - z) +
            + V010 * (1 - x) * y * (1 - z) +
            + V001 * (1 - x) * (1 - y) * z +
            + V101 * x * (1 - y) * z +
            + V011 * (1 - x) * y * z +
            + V110 * x * y * (1 - z) +
            + V111 * x * y * z)
    return Vxyz

def get_intensity(coor, matrix):
    if INTERPOLATION == 'Neigh':
        try:
            coor = [int(round(pos, 0)) for pos in coor]
            return matrix[abs(coor[1]), coor[0], coor[2]] # se interprentan las coordenadas como (y, z, x)
        except: return 0
    if INTERPOLATION == 'Trilinear': 
        try:
            coor = np.array([[abs(coor[1]), coor[0], coor[2]]])
            return int(trilinear(coor, matrix)[0])
        except: return 0
    if INTERPOLATION == 'Tricubic': 
        try:
            coor = np.array([[abs(coor[1]), coor[0], coor[2]]])
            x = np.linspace(0, len(matrix[0][0]), len(matrix[0][0]))
            y = np.linspace(0, len(matrix[0]), len(matrix[0]))
            z = np.linspace(0, len(matrix), len(matrix))
            interp = RegularGridInterpolator((x,y,z), matrix, method="cubic")
            interp = interp(coor)
            return interp
        except: return 0


def get_new_image(matrix, x_angle, y_angle, z_angle, h):
    shape = matrix.shape
    inverse_rotation_matrix = get_rotation_matrix(x_angle, y_angle, z_angle)
    new_image = []
    for j in range(shape[1]):
        line = []
        for i in range(100):
            coor = (inverse_rotation_matrix @ np.array([i, j, h, 0]))[:-1]
            line.append(get_intensity(coor, matrix))
        new_image.append(line)
    return np.array(new_image)

def get_rotation_matrix(x_angle, y_angle, z_angle, inverse=True):
    c = 1
    if inverse: c =-1
    alpha = math.radians(c*x_angle)
    beta = math.radians(c*y_angle)
    gamma = math.radians(c*z_angle)
    rotation_matrix = np.array([
        [cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma), cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma), 0],
        [sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma), sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma), 0],
        [-sin(beta), cos(beta)*sin(gamma), cos(beta)*cos(gamma), 0],
        [0, 0, 0, 1]
        ])
    return rotation_matrix

###################################################


###################################################
#### PARTE B
###################################################

def get_stack(matrix, x_angle, y_angle, z_angle):
    max_depth = 100
    images = []
    for h in range(max_depth):
        images.append(get_new_image(matrix, x_angle, y_angle, z_angle, h))
    return np.dstack(images)

def MIP(stack):
    return np.max(stack, axis=2)

def SUM(stack):
    return np.sum(stack, axis=2)

###################################################





###################################################
### DEPRECATED
###################################################

def get_intersection(shape, point, normal):
    z, y, x = shape
    lines = [Line([0, 0, 0], [x, 0, 0]),
             Line([0, 0, 0], [0, y, 0]),
             Line([0, 0, 0], [0, 0, z]),
             Line([x, 0, 0], [x, y, 0]),
             Line([x, 0, 0], [x, 0, z]),
             Line([x, y, z], [x, 0, z]),
             Line([x, y, z], [x, y, 0]),
             Line([0, y, z], [0, 0, z]),
             Line([0, y, z], [0, y, 0]),
             Line([0, y, z], [x, y, z]),
             Line([x, y, 0], [0, y, 0]),
             Line([0, 0, z], [x, 0, z])]

    plane = Plane(point, normal)
    
    points = set()
    for line in lines:
        try:
            points.add(tuple(plane.intersect_line(line)))
        except:
            pass

    return points

def get_normal(points):
    vector1 = [ j - i for i, j in zip(points[0], points[1])]
    vector2 = [ j - i for i, j in zip(points[0], points[2])]
    normal = Vector(vector1).cross(vector2)
    return list(normal / normal.norm())

def get_range(img):
    shape = img.shape
    rot_matrix = get_rotation_matrix(inverse=False)
    points = (rot_matrix @ np.array([[0, 0, 0], [0, shape[2], 0], [0, 0, shape[1]], [0, 0, 0]]))[:-1].T
    points = [list(map(int, point)) for point in points]
    print(points)
    normal = get_normal(points)
    intersection = get_intersection(shape, points[0], normal)
    print(intersection)
    old_range = np.array([list(inter) + [0] for inter in intersection]).T
    new_range = (rot_matrix @ old_range)[:-1].T
    print(new_range)

    new_range = [list(map(int, point)) for point in new_range]

    print(new_range)

###################################################