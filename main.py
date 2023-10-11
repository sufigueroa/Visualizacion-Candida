import cv2
from PIL import Image
import numpy as np

from params import *
from utils import *

# Leer tiff y convertirlo en np.array
im = read_tiff(PATH)

# Interpolacion isotropica
im = isotropic_interpolation(im)


#################
### Parte A:
#################

# Se calcula la imagen para el plano especificado en los parametros en el archivo params
new = get_new_image(im, X_ANGLE, Y_ANGLE, Z_ANGLE, DEPTH)
save_image(new, f'results/corte-{INTERPOLATION}.png')


# #################
# ### Parte B:
# #################

# Se obtiene el stack de imagenes de distintas profundidades para el corte especificado en los parametros en el archivo params
image_stack = get_stack(im, X_ANGLE, Y_ANGLE, Z_ANGLE)

# Se guarda el valor del MIP para el stack
save_image(MIP(image_stack), f'results/MIP-{INTERPOLATION}.png')

# Se guarda el valor del SUM para el stack
save_image(SUM(image_stack), f'results/SUM-{INTERPOLATION}.png')


#################
### Bonus:
#################

# print("Se comienza a ejecutar el bonus")
# frames = []
# for deg in range(0, 46, 2):
#     print(f"Se encuentra en el angulo {deg}")
#     image_stack = get_stack(im, deg, 0, 0)
#     mip_image = MIP(image_stack)
#     frames.append(mip_image)
# save_video(frames, f'results/MIP-{INTERPOLATION}.avi')

# print("Se termina de ejecutar el bonus")
