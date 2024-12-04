import cv2
import numpy as np
import threading
import time
import os

# Fungsi untuk melakukan convolusi
def partial_convolution(start_row, end_row, target, kernel, result):
    startP = time.time()
    print(f"Thread mulai di {startP:.4f}")
    panKernel, lebKernel = kernel.shape
    
    for i in range(start_row, end_row):
        for j in range(target.shape[1] - lebKernel + 1):
            region = target[i:i + panKernel, j:j + lebKernel]
            result[i, j] = np.sum(region * kernel)
    
        

# Fungsi utama untuk mengonvolusi satu kanal dengan multithreading
def convolve_with_threads(target, kernel, num_threads):
    panGambar, lebGambar = target.shape
    panKernel, lebKernel = kernel.shape
    # print("hrus nya output jdi segini :", panGambar-panKernel+1," x ",lebGambar-lebKernel+1)

    # Matriks output untuk convolusi
    result = np.zeros((panGambar - panKernel + 1, lebGambar - lebKernel + 1), dtype=np.float32)

    # Hitung pembagian tugas
    chunk_size = (panGambar - panKernel + 1) // num_threads
    threads = []

    # Membagi pekerjaan untuk setiap thread
    for i in range(num_threads):
        start_row = i * chunk_size
        end_row = (i + 1) * chunk_size if i != num_threads - 1 else (panGambar - panKernel + 1)
        t = threading.Thread(target=partial_convolution, args=(start_row, end_row, target, kernel, result))
        threads.append(t)
        t.start()
        # print(f"Thread-{i + 1} mulai memproses baris {start_row} hingga {end_row - 1}")
    print("===End Process=-==")

    # Menunggu semua thread selesai
    for t in threads:
        t.join()
    # Normalisasi output 
    output = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return output

# Membaca gambar berwarna (RGB)
image = cv2.imread("refMeme.jpg")


# Kernel Gaussian 5x5
kernel = np.array([[1,  4,  6,  4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],
[4, 16, 24, 16, 4],[1,  4,  6,  4, 1]], dtype=np.float32)

# Jumlah thread
num_threads = os.cpu_count()
print(f"jumlah thread yang bisa di jalankan sebanyak {num_threads}")

# Mencatat waktu mulai
start_time = time.time()

# Convolusi dengan multithreading
merah, hijau, biru = cv2.split(image)
conMerah = convolve_with_threads(merah, kernel, num_threads)
conHijau = convolve_with_threads(hijau, kernel, num_threads)
conBiru = convolve_with_threads(biru, kernel, num_threads)

# Mencatat waktu selesai
end_time = time.time()

# Hitung waktu eksekusi
execution_time = end_time - start_time
print(f"Waktu eksekusi dengan multithreading: {execution_time:.4f} detik")

# Gabungkan hasil menjadi gambar RGB
output_rgb = cv2.merge((conMerah, conHijau, conBiru))

# Tampilkan hasil
cv2.imshow("Original Image", image)
cv2.imshow("Gaussian Blurred RGB", output_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
