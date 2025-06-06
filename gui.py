import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox
from tomograf import loadImage, make_siogram, generate_kernel, convolution_filter, reverse_sinogram, save_dicom, calcualte_mse, normalize
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import pydicom as pydicom
from datetime import datetime




def toggle_dicom_inputs():
    state = "normal" if save_dcm_var.get() else "disabled"
    patient_name_entry.config(state=state)
    patient_id_entry.config(state=state)
    study_date_entry.config(state=state)
    comment_entry.config(state=state)

def process_image():
    detectors_range = int(detectors_range_combo.get())
    detectors_num = int(detectors_num_combo.get())
    scans = int(scans_combo.get())
    interval = 360 / scans
    use_filter = filter_var.get()
    save_as_dcm = save_dcm_var.get()
    save_as_jpg = save_jpg_var.get()
    show_steps = show_steps_var.get()

    file_path = image_path.get()
    try:
        if file_path[-4:] == ".dcm":
            dcm = pydicom.dcmread("output.dcm")
            og_img = normalize(dcm.pixel_array)
            ds = pydicom.dcmread(file_path)
            p_name = dcm.PatientName
            p_id = dcm.get('PatientID', 'NONE')
            date = datetime.strptime(dcm.get('StudyDate', ''), '%Y%m%d').date()
            p_comment = dcm.get('PatientComments', 'NONE')
        else:
            og_img = loadImage(image_path.get())
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")
        return

    try:
        sinogram = make_siogram(
            og_img, interval, detectors_range=detectors_range, detectors_num=detectors_num,
            tk_canvas=canvas if show_steps else None, update_interval=10
        )
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate sinogram: {e}")
        return

    if use_filter:
        kernel = generate_kernel(21)
        sinogram = convolution_filter(sinogram, kernel)

        try:
            normalized_sinogram = normalize(sinogram)
            sinogram_image = Image.fromarray(normalized_sinogram.astype(np.uint8))
            sinogram_photo = ImageTk.PhotoImage(sinogram_image)
            filtered_sinogram_canvas.create_image(0, 0, anchor=tk.NW, image=sinogram_photo)
            filtered_sinogram_canvas.image = sinogram_photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display filtered sinogram: {e}")
            return
        
    try:
        reconstructed_image = reverse_sinogram(sinogram, og_img, interval=interval, detectors_range=detectors_range, detectors_num=detectors_num)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to reverse sinogram: {e}")
        return

    reconstructed_image = reconstructed_image.astype('uint8')

    # try:
    #     reconstructed_image_pil = Image.fromarray(reconstructed_image)
    #     reconstructed_image_photo = ImageTk.PhotoImage(reconstructed_image_pil)
    #     reconstructed_image_canvas.create_image(0, 0, anchor=tk.NW, image=reconstructed_image_photo)
    #     reconstructed_image_canvas.image = reconstructed_image_photo
    # except Exception as e:
    #     messagebox.showerror("Error", f"Failed to display reconstructed image: {e}")
    #     return
    

    if save_as_dcm:
        patient_name = patient_name_var.get()
        patient_id = patient_id_var.get()
        study_date = study_date_var.get()
        comment = comment_var.get()
        try:
            save_dicom(reconstructed_image, filename="output.dcm", patient_name=patient_name, patient_id=patient_id, study_date=study_date, comment=comment)

            ds = pydicom.dcmread("output.dcm")
            messagebox.showinfo("Success", "Reconstructed image saved as output.dcm")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save DICOM: {e}")
            return

    if save_as_jpg:
        try:
            rec_img = Image.fromarray(reconstructed_image)
            rec_img.save("output.jpg")
            messagebox.showinfo("Success", "Reconstructed image saved as output.jpg")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save JPG: {e}")
            return

    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis("off")
    mse, rmse = calcualte_mse(og_img,rec_img)
    if file_path[-4:] == ".dcm":
        patient_info = [
            f"Patient name: {p_name}",
            f"Date: {date}",
            f"ID: {p_id}",
            f"Comment: {p_comment}"
        ]

        y_start = 0.15
        line_spacing = 0.03
        last_y = y_start

        for i, info in enumerate(patient_info):
            y_pos = y_start - i * line_spacing
            plt.figtext(0.5, y_pos, info, wrap=True,
                        horizontalalignment='center', fontsize=10)
            last_y = y_pos

        rmse_y = last_y - 0.04
        plt.figtext(0.4, rmse_y, f"MSE = {mse:.4f}", wrap=True,
                    horizontalalignment='right', fontsize=10)
        plt.figtext(0.6, rmse_y, f"RMSE = {rmse:.4f}", wrap=True,
                    horizontalalignment='left', fontsize=10)

        plt.subplots_adjust(bottom=0.3)
    else:
        plt.figtext(0.3,0.01, f"MSE = {mse}", wrap=True, horizontalalignment='center', fontsize=10)
        plt.figtext(0.7,0.01, f"RMSE = {rmse}", wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp"),("Dicom","*.dcm")])
    if file_path:
        image_path.set(file_path)

root = tk.Tk()
root.title("Tomograf GUI")

image_path = tk.StringVar()
tk.Label(root, text="Select Image:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
tk.Entry(root, textvariable=image_path, width=40).grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2, padx=10, pady=10)

tk.Label(root, text="Rozpiętość wachlarza:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
detectors_range_combo = Combobox(root, values=[45, 90, 135, 180, 225, 270], state="readonly")
detectors_range_combo.current(0)
detectors_range_combo.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Liczba detektorów:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
detectors_num_combo = Combobox(root, values=[90, 180, 270, 360, 450, 540, 630, 720], state="readonly")
detectors_num_combo.current(0)
detectors_num_combo.grid(row=2, column=1, padx=10, pady=10)

tk.Label(root, text="Liczba skanów:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
scans_combo = Combobox(root, values=[90, 180, 270, 360, 450, 540, 630, 720], state="readonly")
scans_combo.current(0)
scans_combo.grid(row=3, column=1, padx=10, pady=10)

filter_var = tk.BooleanVar()
tk.Checkbutton(root, text="Use Convolution Filter", variable=filter_var).grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="w")

save_dcm_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Save as DICOM", variable=save_dcm_var, command=toggle_dicom_inputs).grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="w")

save_jpg_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Save as JPG", variable=save_jpg_var).grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="w")

tk.Label(root, text="Patient Name:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
patient_name_var = tk.StringVar()
patient_name_entry = tk.Entry(root, textvariable=patient_name_var, width=30)
patient_name_entry.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Patient ID:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
patient_id_var = tk.StringVar()
patient_id_entry = tk.Entry(root, textvariable=patient_id_var, width=30)
patient_id_entry.grid(row=8, column=1, padx=10, pady=5)

tk.Label(root, text="Study Date (YYYY/MM/DD):").grid(row=9, column=0, padx=10, pady=5, sticky="w")
study_date_var = tk.StringVar()
study_date_entry = tk.Entry(root, textvariable=study_date_var, width=30)
study_date_entry.grid(row=9, column=1, padx=10, pady=5)

tk.Label(root, text="Comment:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
comment_var = tk.StringVar()
comment_entry = tk.Entry(root, textvariable=comment_var, width=30)
comment_entry.grid(row=10, column=1, padx=10, pady=5)

show_steps_var = tk.BooleanVar()
tk.Checkbutton(root, text="Show Intermediate Steps", variable=show_steps_var).grid(row=11, column=0, columnspan=2, padx=10, pady=10, sticky="w")

canvas = tk.Canvas(root, width=400, height=400, bg="white")
canvas.grid(row=0, column=3, rowspan=14, padx=10, pady=10, sticky="n")

filtered_sinogram_canvas = tk.Canvas(root, width=400, height=400, bg="white")
filtered_sinogram_canvas.grid(row=0, column=4, rowspan=7, padx=10, pady=10, sticky="n")

# reconstructed_image_canvas = tk.Canvas(root, width=400, height=400, bg="white")
# reconstructed_image_canvas.grid(row=7, column=4, rowspan=7, padx=10, pady=10, sticky="n")

tk.Button(root, text="Process", command=process_image).grid(row=12, column=0, columnspan=3, pady=20)

toggle_dicom_inputs()

if __name__ == "__main__":
    root.mainloop()