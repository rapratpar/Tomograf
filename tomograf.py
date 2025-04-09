import math
from tkinter import Tk
import tkinter
import skimage as sk
import numpy as np
import pydicom as dic
import datetime
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import pydicom as pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
#import pydicom._storage_sopclass_uids
import datetime


from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity


def convert_image_to_ubyte(img):
    return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))


# Normalizacja do przedzialu 0-255 dla nieujemnych wartosci
def normalize(image):
    norm = np.zeros(shape=(image.shape))
    image_max = image.max()
    image_min = image.min()
    print(image_max, image_min)

    for x in range(image.shape[0]):
        # Przekształcenie wartosci
        norm[x] = np.interp(image[x], (image_min, image_max), (0, 255))
    return norm



# Ładownia obrazu w skali szarosci i normalizacja
def loadImage(image_path): # ładuje obrazek
    image = sk.io.imread(image_path, as_gray=True)
    image = normalize(image)
    return image

# Jeśli są ujemne wartości, przeskalowanie do 0-255 i zapisanie jako całkowitoliczbowe (Z jakiegoś powodu zwykła normalizacja nie działa z dicom)
def scale_img(image):
    image_scaled = (image - image.min()) / (image.max() - image.min()) * 255

    # Konwersja do typu całkowitego (8-bitowego)
    return image_scaled.astype(np.uint8)

# Funkcja do znalezienia punktów między emiterm i dekoderem
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


# Stoworzenie kernela do filtracji (21 jako domyślny, srodkowa wartość 0, wystarczy obliczyć połowe druga jest analogicznie)
def generate_kernel(size: int = 21) -> np.ndarray:
    if size % 2 == 0 or size < 1:
        return np.array([1])
    middle = size // 2
    filter_kernel = np.ones(size)
    for k in range(1, size // 2 + 1):
        if k % 2 == 0:
            filter_kernel[middle - k] = filter_kernel[middle + k] = 0
        else:
            filter_kernel[middle - k] = filter_kernel[middle + k] = (-4 / (np.pi ** 2)) / (k ** 2)
    filter_kernel[middle] = 1
    #print(filter_kernel)
    return filter_kernel

# Filtorwanie sinogramu
def convolution_filter(sinogram: np.ndarray, filter_kernel: np.array) -> np.array:
    filtered_sinogram = np.zeros_like(sinogram)
    for i, row in enumerate(sinogram):
        # Wykonuje splot syngałow jednokanałowych
        filtered_row = np.convolve(sinogram[i, :], filter_kernel, mode='same')
        filtered_sinogram[i, :] = filtered_row
    return filtered_sinogram
def make_siogram(img, interval, detectors_range, detectors_num, tk_canvas=None, update_interval=10):
    """

    TODO: Dodać iterracyjne podejśćie tzn: W tkinterze zamiast czekać na cały wynik, aplikacja będzie co jakiś czas
    wyświetlać użytkownikowi postęp tworzenia sinogramu (Aktulanie wypełniony)
    Trzeba będzie przekazać tutaj jakiś obiekt tkintera który będzie ładować

    :param img: Obraz który przetwarzamy
    :param interval: Krok z jakim emiter przesuwa się po okręgu
    :param detectors_range: Kąt rozpiętości detectorów dla jednego pomiaru
    :param detectors_num: Ilość detectorów w jednym pomiarze
    :return: Zwraca znormalizowany sinogram
    """


    # Tablica z kątami emitera w radianach
    emitter_angles = np.deg2rad(np.arange(0,360,interval))

    # Srodek obrazka
    X,Y = img.shape[0] // 2,img.shape[1] // 2

    # Bierzemiy mniejszy długość środka jako promień
    R = min(X,Y) * math.sqrt(2)

    # Rozmiary sinogramu: ilość projekci x ilość detektorów
    sinogram = np.zeros((len(emitter_angles),detectors_num))

    detectors_range_rad = np.deg2rad(detectors_range)

    # Przejscie po wszystkich pozycjach emitera
    for emitter_id, emitter_angle in enumerate(emitter_angles):
        #print(emitter_id)
        # Oblicznie coordów emitera
        emitter_x,emitter_y = calcualte_emitter_pos(X,Y,R,emitter_angle)
        emitter_x, emitter_y = int(emitter_x), int(emitter_y)

        # Znajdujemy skrajne wartosci dekodorów, i ustuwaimy reszte dekoderów międzey nimi
        detector_end,detector_start = (emitter_angle + np.pi + (detectors_range_rad/2), emitter_angle + np.pi - (detectors_range_rad/2))
        detectors_angels = np.linspace(detector_start,detector_end,detectors_num)

        # Przechodzimy przez kazdy dekoder
        for detector_id, detector_angele in enumerate(detectors_angels):
            #Wyznaczenie pozycji dekodera
            detector_x,detector_y =  (X + np.round(R*np.cos(detector_angele)),Y + np.round(R*np.sin(detector_angele)))
            detector_x, detector_y = int(detector_x), int(detector_y)

            # Wyliczenie punktów miedzy emiterm i dekoderm, oraz wartosci na tej lini
            sinogram[emitter_id][detector_id] = bresenham_line(img, emitter_x, emitter_y, detector_x, detector_y)

        if tk_canvas and emitter_id % update_interval == 0:
            normalized_sinogram = normalize(sinogram)
            tk_image = Image.fromarray(normalized_sinogram.astype(np.uint8))
            tk_photo = ImageTk.PhotoImage(tk_image)
            tk_canvas.create_image(0, 0, anchor=tkinter.NW, image=tk_photo)
            tk_canvas.image = tk_photo
            tk_canvas.update()
            
    return normalize(sinogram)

def reverse_sinogram(sinogram,img,interval,detectors_range,detectors_num):
    """
    To dziala analogicznie jak sinogram, róznica taka że tutaj alorytm bersehama zwraca punkty na onbrazku na których ustawaimy
    wartosci z danej iteracji emiter-dekoder (defacto odwracamy proces)
    """

    # Wyliczamy pozycje emitera na okregu
    emitter_angles = np.deg2rad(np.arange(0, 360, interval))
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


# Wzory z pdf z ekursow
def calcualte_emitter_pos(x,y,r,emitter_pos):
    return  (x+np.round(r * np.cos(emitter_pos)), y + np.round(r * np.sin(emitter_pos)))



def save_dicom(image_array, filename="output.dcm", patient_name="Unknown", patient_id="0", study_date="", comment="No comment"):
    """ Tworzy plik DICOM z macierzy pikseli """
    #print(image_array.dtype)
    # Jeśli obraz jest w zakresie 0-1 (float), normalizujemy do 0-255 (uint8)
    #if image_array.dtype == np.float32 or image_array.dtype == np.float64:
    #    image_array = (image_array * 255).astype(np.uint8)
    #img_converted = convert_image_to_ubyte(image_array)
    # 📌 1️⃣ Tworzenie nagłówka metadanych
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID("1.2.840.10008.5.1.4.1.1.2")  # SOP Class UID dla CT
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.UID("1.2.826.0.1.3680043.8.498.1")  # Unikalne ID implementacji
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  # 🔴 POPRAWKA! Ustawienie Transfer Syntax UID

    # 📌 2️⃣ Tworzenie głównego obiektu DICOM
    dicom_ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # 📌 3️⃣ Generowanie unikalnych identyfikatorów
    dicom_ds.SOPInstanceUID = pydicom.uid.generate_uid()
    dicom_ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    dicom_ds.StudyInstanceUID = pydicom.uid.generate_uid()
    dicom_ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    # 📌 4️⃣ Dane pacjenta
    dt = datetime.datetime.now()
    dicom_ds.StudyDate = study_date if study_date else dt.strftime('%Y%m%d')
    dicom_ds.ContentDate = study_date if study_date else dt.strftime('%Y%m%d')
    dicom_ds.StudyTime = dt.strftime('%H%M%S')
    dicom_ds.PatientName = patient_name
    dicom_ds.PatientID = patient_id
    dicom_ds.StudyID = "1234"
    dicom_ds.SeriesNumber = "1"
    dicom_ds.PatientComments = comment

    # 📌 5️⃣ Ustawienia obrazu
    dicom_ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    dicom_ds.Modality = "CT"
    dicom_ds.Rows, dicom_ds.Columns = image_array.shape
    dicom_ds.BitsAllocated = 8
    dicom_ds.BitsStored = 8
    dicom_ds.HighBit = dicom_ds.BitsStored - 1
    dicom_ds.SamplesPerPixel = 1
    dicom_ds.PhotometricInterpretation = "MONOCHROME2"
    dicom_ds.PixelRepresentation = 0
    dicom_ds.PixelData = image_array.tobytes()

    # 📌 6️⃣ Ustawienie sposobu kodowania danych
    dicom_ds.is_little_endian = True
    dicom_ds.is_implicit_VR = False  # Explicit VR zapewnia lepszą kompatybilność

    # 📌 7️⃣ Zapisywanie pliku DICOM
    dicom_ds.save_as(filename, write_like_original=False)
    print(f"Plik DICOM zapisano jako {filename}")

    return filename

def open_dicom(filename):
    """ Otwiera plik DICOM i wyświetla obraz oraz jego metadane """
    dicom_ds = pydicom.dcmread(filename)

    # Wyświetlenie podstawowych informacji
    print(f"Pacjent: {dicom_ds.PatientName}")
    print(f"ID pacjenta: {dicom_ds.PatientID}")
    print(f"Data badania: {dicom_ds.StudyDate}")
    print(f"Komentarz: {dicom_ds.PatientComments}")

    # Wyświetlenie obrazu
    image_array = dicom_ds.pixel_array
    #print(image_array[0])
    plt.imshow(image_array, cmap="gray")
    plt.title("Obraz DICOM")
    plt.axis("off")
    plt.show()


# img = loadImage("Kropka.jpg")

# sinogram = make_siogram(img,1, 270,360)
# kernel = generate_kernel(21)
# plt.imshow(sinogram, cmap='gray', aspect='auto')  # 'gray' to mapa kolorów do wyświetlania obrazów w skali szarości
# plt.title('Sinogram')
# plt.colorbar()  # Dodanie paska kolorów, żeby zobaczyć wartości
# plt.axis('off')  # Ukrycie osi, jeżeli nie są potrzebne
# plt.show()

# f_sinogram = convolution_filter(sinogram, kernel)
# plt.imshow(f_sinogram, cmap='gray', aspect='auto')  # 'gray' to mapa kolorów do wyświetlania obrazów w skali szarości
# plt.title('Sinogram')
# plt.colorbar()  # Dodanie paska kolorów, żeby zobaczyć wartości
# plt.axis('off')  # Ukrycie osi, jeżeli nie są potrzebne
# plt.show()

# r_s = reverse_sinogram(f_sinogram,img,1, 270,360)

# plt.imshow(r_s, cmap='gray', aspect='auto')  # 'gray' to mapa kolorów do wyświetlania obrazów w skali szarości
# plt.title('Sinogram')
# plt.colorbar()  # Dodanie paska kolorów, żeby zobaczyć wartości
# plt.axis('off')  # Ukrycie osi, jeżeli nie są potrzebne
# plt.show()
# #print(r_s[0])
# r_s = normalize(r_s)
# #print(r_s[0])
# # img to twoja tablica obrazu
# img_scaled = (r_s - r_s.min()) / (r_s.max() - r_s.min()) * 255
# # Konwersja do typu całkowitego (8-bitowego) ( Z jakieos powodu bez tego zapisuje źle w dicom, prawdopodobnie zła konwersja z float na unit8 w normalizacji)
# img_scaled = img_scaled.astype(np.uint8)

# #print(img_scaled[0])

# save_dicom(img_scaled)

# #Sprawdze czy obraz dicom dobrze się zapisuje
# open_dicom("output.dcm")