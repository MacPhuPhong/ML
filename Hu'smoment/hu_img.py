import cv2
import numpy as np

# --- 1. Đọc ảnh ---
img = cv2.imread("//media/pphong/New Volume/ML & DL/ML/Hu'smoment/dataset_leafs_binary/La_Trau_5.jpg", cv2.IMREAD_GRAYSCALE)

# --- 2. Chuyển về 0/1 ---
binary = (img > 127).astype(np.uint8)

h, w = binary.shape
y, x = np.mgrid[0:h, 0:w]

# --- 3. Raw moments ---
def raw_moment(p, q):
    return (x**p * y**q * binary).sum()

m00 = raw_moment(0, 0)
m10 = raw_moment(1, 0)
m01 = raw_moment(0, 1)

# --- 4. Tâm ---
x_bar = m10 / m00
y_bar = m01 / m00

# --- 5. Central moments ---
def central_moment(p, q):
    return (((x - x_bar)**p) * ((y - y_bar)**q) * binary).sum()

mu20 = central_moment(2, 0)
mu02 = central_moment(0, 2)
mu11 = central_moment(1, 1)
mu30 = central_moment(3, 0)
mu03 = central_moment(0, 3)
mu12 = central_moment(1, 2)
mu21 = central_moment(2, 1)

# --- 6. Normalized central moments ---
def eta(mu, p, q):
    return mu / (m00 ** (1 + (p + q)/2))

eta20 = eta(mu20, 2, 0)
eta02 = eta(mu02, 0, 2)
eta11 = eta(mu11, 1, 1)
eta30 = eta(mu30, 3, 0)
eta03 = eta(mu03, 0, 3)
eta12 = eta(mu12, 1, 2)
eta21 = eta(mu21, 2, 1)

# --- 7. Hu moments ---
phi1 = eta20 + eta02
phi2 = (eta20 - eta02)**2 + 4*(eta11**2)
phi3 = (eta30 - 3*eta12)**2 + (3*eta21 - eta03)**2
phi4 = (eta30 + eta12)**2 + (eta21 + eta03)**2
phi5 = (eta30 - 3*eta12)*(eta30 + eta12)*((eta30+eta12)**2 - 3*(eta21+eta03)**2) + \
       (3*eta21 - eta03)*(eta21+eta03)*(3*(eta30+eta12)**2 - (eta21+eta03)**2)
phi6 = (eta20 - eta02)*((eta30+eta12)**2 - (eta21+eta03)**2) + \
       4*eta11*(eta30+eta12)*(eta21+eta03)
phi7 = (3*eta21 - eta03)*(eta30+eta12)*((eta30+eta12)**2 - 3*(eta21+eta03)**2) - \
       (eta30 - 3*eta12)*(eta21+eta03)*(3*(eta30+eta12)**2 - (eta21+eta03)**2)

print("Hu Moments tính tay từ ảnh lá:")
print("φ1 =", phi1)
print("φ2 =", phi2)
print("φ3 =", phi3)
print("φ4 =", phi4)
print("φ5 =", phi5)
print("φ6 =", phi6)
print("φ7 =", phi7)
