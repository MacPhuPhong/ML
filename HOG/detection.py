import cv2
import numpy as np
from numpy.linalg import norm

# -------------------------------
# 1. Hàm tính HOG thủ công
# -------------------------------
def compute_hog(img, cell_size=8, block_size=2, bins=9):
    # B1: Gradient theo X, Y
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 180  # góc 0-180
    
    h, w = img.shape
    cell_h, cell_w = cell_size, cell_size
    n_cells_x = w // cell_w
    n_cells_y = h // cell_h
    
    # B2: Histogram cho từng cell
    hist = np.zeros((n_cells_y, n_cells_x, bins))
    bin_width = 180 // bins
    
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_angle = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            for y in range(cell_mag.shape[0]):
                for x in range(cell_mag.shape[1]):
                    mag = cell_mag[y, x]
                    ang = cell_angle[y, x]
                    bin_idx = int(ang // bin_width) % bins
                    hist[i, j, bin_idx] += mag
    
    # B3: Block normalization (2x2 cell)
    eps = 1e-6
    hog_vector = []
    for i in range(n_cells_y - block_size + 1):
        for j in range(n_cells_x - block_size + 1):
            block = hist[i:i+block_size, j:j+block_size, :].ravel()
            block_norm = block / (np.sqrt(np.sum(block**2)) + eps)
            hog_vector.extend(block_norm)
    
    return np.array(hog_vector)

# -------------------------------
# 2. Template HOG
# -------------------------------
template = cv2.imread("leaf_template.png", cv2.IMREAD_GRAYSCALE)
template = cv2.resize(template, (64, 128))  # chuẩn hóa size
fd_template = compute_hog(template)

# -------------------------------
# 3. So khớp bằng sliding window
# -------------------------------
def sliding_window(img, step_size, window_size):
    for y in range(0, img.shape[0] - window_size[1], step_size):
        for x in range(0, img.shape[1] - window_size[0], step_size):
            yield (x, y, img[y:y + window_size[1], x:x + window_size[0]])

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (norm(a) * norm(b))

# def manhattan_distance(a, b):
#     return np.sum(np.abs(a - b))

test_img = cv2.imread("scene.png", cv2.IMREAD_GRAYSCALE)
window_size = (64, 128)
detections = []

for (x, y, window) in sliding_window(test_img, step_size=16, window_size=window_size):
    if window.shape != (window_size[1], window_size[0]):
        continue
    fd_win = compute_hog(window)
    sim = cosine_similarity(fd_template, fd_win)
    if sim > 0.85:   # ngưỡng so khớp
        detections.append((x, y, x+window_size[0], y+window_size[1], sim))

# -------------------------------
# 4. Vẽ kết quả
# -------------------------------
output = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
for (x1, y1, x2, y2, s) in detections:
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(output, f"{s:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0,255,0), 1)

cv2.imwrite("detections_hog_manual.png", output)
print("✅ Hoàn tất: kết quả lưu ở detections_hog_manual.png")

# def manhattan_distance(a, b):
#     return np.sum(np.abs(a - b))

