import math
import skimage as sk
import numpy as np
import pydicom as dic
import datetime
from PIL import Image
from matplotlib import pyplot as plt


def normalize(im): # normalizacja obrazu
  norm = np.zeros(shape=(im.shape))
  im_max = im.max()
  im_min = 0
  for x in range(im.shape[0]):
    norm[x] = np.interp(im[x], (im_min, im_max), (0, 255))
  return norm

def loadImage(imgPath): # ładuje obrazek
  im = sk.io.imread(imgPath, as_gray=True)
  im = normalize(im)
  return im


def bresenham_line(img, x1, y1, x2, y2, return_coords=False):
    max_x, max_y = img.shape
    coords = []
    pixel_values = []

    x, y = x1, y1
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    x_inc = 1 if x2 > x1 else -1
    y_inc = 1 if y2 > y1 else -1

    if dx > dy:  # Oś wiodąca OX
        ep = 2 * dy - dx
        for _ in range(dx + 1):
            if 0 <= x < max_x and 0 <= y < max_y:
                coords.append((x, y))
                pixel_values.append(img[x, y])
            if ep >= 0:
                y += y_inc
                ep -= 2 * dx
            ep += 2 * dy
            x += x_inc
    else:  # Oś wiodąca OY
        ep = 2 * dx - dy
        for _ in range(dy + 1):
            if 0 <= x < max_x and 0 <= y < max_y:
                coords.append((x, y))
                pixel_values.append(img[x, y])
            if ep >= 0:
                x += x_inc
                ep -= 2 * dy
            ep += 2 * dx
            y += y_inc

    if return_coords:
        return coords  # Zwracamy listę współrzędnych pikseli
    return np.mean(pixel_values) if pixel_values else 0  # Zwracamy średnią jasność pikseli



def generate_kernel(size: int) -> np.array:
    if size % 2 == 0 or size < 1:
        return np.array([1])
    middle = size // 2
    filter_kernel = np.ones(size)
    for k in range(1, size // 2 + 1):
        if k % 2 == 0:
            filter_kernel[middle - k] = filter_kernel[middle + k] = 0
        else:
            filter_kernel[middle - k] = filter_kernel[middle + k] = (-4 / (np.pi ** 2)) / (k ** 2)
    return filter_kernel

def convolution_filter(sinogram: np.ndarray, filter_kernel: np.array) -> np.array:
    filtered_sinogram = np.zeros_like(sinogram)
    for i, row in enumerate(sinogram):
        filtered_row = np.convolve(sinogram[i, :], filter_kernel, mode='same')
        filtered_sinogram[i, :] = filtered_row
    return filtered_sinogram
def make_siogram(img, interval, detectors_range, detectors_num):
    # Wyliczamy pozycje emitera na okregu
    emitter_angles = np.deg2rad(np.arange(90,450,interval))
    # Znajduejmy srodek obrazka
    X,Y = img.shape[0] // 2,img.shape[1] // 2
    # Bierzemiy mniejszy długość środka jako promień
    R = min(X,Y) *math.sqrt(2)
    sinogram = np.zeros((len(emitter_angles),detectors_num))
    detectors_range_rad = np.deg2rad(detectors_range)

    for emitter_id, emitter_angle in enumerate(emitter_angles):
        #print(emitter_id)
        emitter_x,emitter_y = calcualte_emitter_pos(X,Y,R,emitter_angle)
        emitter_x, emitter_y = int(emitter_x), int(emitter_y)
        detector_end,detector_start = (emitter_angle + np.pi + (detectors_range_rad/2), emitter_angle + np.pi - (detectors_range_rad/2))
        detectors_angels = np.linspace(detector_start,detector_end,detectors_num)
        for detector_id, detector_angele in enumerate(detectors_angels):
            detector_x,detector_y =  (X + np.round(R*np.cos(detector_angele)),Y + np.round(R*np.sin(detector_angele)))
            detector_x, detector_y = int(detector_x), int(detector_y)

            #sinogram[emitter_id][detector_id] = bresenham(img, emitter_x, emitter_y, detector_x, detector_y)
            sinogram[emitter_id][detector_id] = bresenham_line(img, emitter_x, emitter_y, detector_x, detector_y)
    return normalize(sinogram)

def reverse_sinogram(sinogram,img,interval,detectors_range,detectors_num):
    # Wyliczamy pozycje emitera na okregu
    emitter_angles = np.deg2rad(np.arange(90, 450, interval))
    # Znajduejmy srodek obrazka
    X, Y = img.shape[0] // 2, img.shape[1] // 2
    # Promien na podstawie mniejszego boku
    R = min(X, Y) * math.sqrt(2)
    reversed_sinogram = np.zeros(shape=img.shape)
    detectors_range_rad = np.deg2rad(detectors_range)

    for emitter_id, emitter_angle in enumerate(emitter_angles):
        emitter_x, emitter_y = calcualte_emitter_pos(X, Y, R, emitter_angle)
        emitter_x, emitter_y = int(emitter_x), int(emitter_y)
        detector_end, detector_start = (
        emitter_angle + np.pi + (detectors_range_rad / 2), emitter_angle + np.pi - (detectors_range_rad / 2))
        detectors_angels = np.linspace(detector_start, detector_end, detectors_num)
        for detector_id, detector_angele in enumerate(detectors_angels):
            detector_x, detector_y = (X + np.round(R * np.cos(detector_angele)), Y + np.round(R * np.sin(detector_angele)))
            detector_x, detector_y = int(detector_x), int(detector_y)
            #coords = inverseBresenham(img.shape[0], img.shape[1],emitter_x, emitter_y, detector_x, detector_y)
            coords = bresenham_line(img, emitter_x, emitter_y, detector_x, detector_y, True)

              # wzmacniamy piksele wyznaczonej linii o odpowiednią średnią
            reversed_sinogram [tuple(np.transpose(coords))] += sinogram[emitter_id][detector_id]

    return normalize(reversed_sinogram)


def calcualte_emitter_pos(x,y,r,emitter_pos):
    return  (x+np.round(r * np.cos(emitter_pos)), y + np.round(r * np.sin(emitter_pos)))



img = loadImage("Kropka.jpg")

sinogram = make_siogram(img,1, 180,180)
kernel = generate_kernel(21)
plt.imshow(sinogram, cmap='gray', aspect='auto')  # 'gray' to mapa kolorów do wyświetlania obrazów w skali szarości
plt.title('Sinogram')
plt.colorbar()  # Dodanie paska kolorów, żeby zobaczyć wartości
plt.axis('off')  # Ukrycie osi, jeżeli nie są potrzebne
plt.show()

f_sinogram = convolution_filter(sinogram, kernel)
plt.imshow(f_sinogram, cmap='gray', aspect='auto')  # 'gray' to mapa kolorów do wyświetlania obrazów w skali szarości
plt.title('Sinogram')
plt.colorbar()  # Dodanie paska kolorów, żeby zobaczyć wartości
plt.axis('off')  # Ukrycie osi, jeżeli nie są potrzebne
plt.show()

r_s = reverse_sinogram(f_sinogram,img,1, 180,180)

plt.imshow(r_s, cmap='gray', aspect='auto')  # 'gray' to mapa kolorów do wyświetlania obrazów w skali szarości
plt.title('Sinogram')
plt.colorbar()  # Dodanie paska kolorów, żeby zobaczyć wartości
plt.axis('off')  # Ukrycie osi, jeżeli nie są potrzebne
plt.show()