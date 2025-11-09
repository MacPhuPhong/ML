import cv2
import numpy as np

# --- 1. Load ảnh và resize ---
img_path = r"/media/pphong/New Volume/ML & DL/ML/HOG/hinh-cho-hai-husky-ngam-bong.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)   # đọc ảnh grayscale
img = cv2.resize(img, (64, 128))                   # resize về chuẩn HOG: 64x128

# --- 2. Tính gradient thủ công ---
# 1. Tạo kernel Sobel thủ công
Sx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)

Sy = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=np.float32)

# 2. Convolution (tích chập) để tính Gx, Gy
Gx = cv2.filter2D(img.astype(np.float32), -1, Sx)   # gradient theo trục x
Gy = cv2.filter2D(img.astype(np.float32), -1, Sy)   # gradient theo trục y

magnitude = np.sqrt(Gx**2 + Gy**2)                 # độ lớn vector gradient M = sqrt(Gx^2+Gy^2)
angle = np.arctan2(Gy, Gx) * 180 / np.pi           # tính góc gradient (radian -> độ)
angle[angle < 0] += 180                            # chuẩn hóa về khoảng [0,180) (unsigned)

# --- 3. Chia cell 8x8 và tính histogram ---
cell_size = 8                                      # kích thước cell = 8x8 pixels
nbins = 9                                          # số bins của histogram = 9
bin_width = 180 // nbins                           # độ rộng mỗi bin = 20°

n_cells_x = img.shape[1] // cell_size              # số cell theo chiều ngang = 64/8 = 8
n_cells_y = img.shape[0] // cell_size              # số cell theo chiều dọc = 128/8 = 16

# mảng lưu histogram cho từng cell, kích thước (16, 8, 9)
hist_cells = np.zeros((n_cells_y, n_cells_x, nbins)) 

for i in range(n_cells_y):                         # duyệt cell theo chiều dọc
    for j in range(n_cells_x):                     # duyệt cell theo chiều ngang
        # lấy magnitude và angle trong cell hiện tại
        cell_mag = magnitude[i*cell_size:(i+1)*cell_size,
                             j*cell_size:(j+1)*cell_size]  
        cell_angle = angle[i*cell_size:(i+1)*cell_size,
                           j*cell_size:(j+1)*cell_size]    

        for u in range(cell_size):                 # duyệt từng pixel trong cell
            for v in range(cell_size):
                a = cell_angle[u, v]               # góc gradient tại pixel
                m = cell_mag[u, v]                 # độ lớn gradient tại pixel
                bin_idx = int(a // bin_width)      # bin gần nhất bên trái
                if bin_idx >= nbins:               # FIX: nếu góc =180° thì bin_idx = 9 -> vượt mảng
                    bin_idx = nbins - 1            # ép về bin cuối (bin số 8)
                next_bin = (bin_idx + 1) % nbins   # bin kế tiếp bên phải (dùng modulo để vòng tròn)
                ratio = (a - bin_idx*bin_width) / bin_width   # tỉ lệ nội suy tuyến tính
                hist_cells[i, j, bin_idx] += m * (1 - ratio)  # cộng magnitude * tỉ lệ vào bin trái
                hist_cells[i, j, next_bin] += m * ratio       # cộng magnitude * tỉ lệ vào bin phải

# --- 4. Gom cell thành block và chuẩn hoá ---
block_size = 2                                     # block = 2x2 cells
block_features = []                                # danh sách vector đặc trưng block

for i in range(n_cells_y - block_size + 1):        # duyệt block theo chiều dọc
    for j in range(n_cells_x - block_size + 1):    # duyệt block theo chiều ngang
        # lấy histogram 4 cell (2x2), rồi nối lại thành vector 36 chiều
        block = hist_cells[i:i+block_size, j:j+block_size, :].ravel()  
        norm = np.linalg.norm(block) + 1e-5        # chuẩn hoá L2 (cộng epsilon để tránh chia 0)
        block = block / norm                       # vector sau chuẩn hoá
        block_features.append(block)               # thêm block vào danh sách

# nối toàn bộ block -> vector HOG
hog_vector = np.concatenate(block_features)        

print("Chiều dài vector HOG:", hog_vector.shape[0]) # in số chiều HOG (chuẩn = 3780)
print("Vector HOG:", hog_vector)                     # in vector HOG
