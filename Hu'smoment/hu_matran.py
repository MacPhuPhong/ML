import numpy as np

# --- 1. Tạo ma trận ảnh nhị phân 4x4 ---
# Ở đây tạo hình vuông 2x2
img = np.array([[0,0,0,0],
                [0,1,1,0],
                [0,1,1,0],
                [0,1,0,0]])

print("Ảnh nhị phân:\n", img)

# --- 2. Tính raw moments ---
def raw_moment(img, p, q):
    m = 0
    for (y, x), val in np.ndenumerate(img):
        m += (x**p) * (y**q) * val
    return m

m00 = raw_moment(img, 0, 0)
m10 = raw_moment(img, 1, 0)

m01 = raw_moment(img, 0, 1)

print("\nm00 =", m00, " (diện tích)")
print("m10 =", m10)
print("m01 =", m01)

# --- 3. Centroid ---
x_bar = m10 / m00
y_bar = m01 / m00
print("\nTrọng tâm (x̄, ȳ) =", (x_bar, y_bar))

# --- 4. Central moments ---
def central_moment(img, p, q, x_bar, y_bar):
    mu = 0
    for (y, x), val in np.ndenumerate(img):
        mu += ((x - x_bar)**p) * ((y - y_bar)**q) * val
    return mu

mu20 = central_moment(img, 2, 0, x_bar, y_bar)
mu02 = central_moment(img, 0, 2, x_bar, y_bar)
mu11 = central_moment(img, 1, 1, x_bar, y_bar)
mu22 = central_moment(img, 2, 2, x_bar, y_bar)

print("\nμ20 =", mu20)
print("μ02 =", mu02)
print("μ11 =", mu11)
print("μ22 =", mu22)

# --- 5. Normalized central moments ---
eta20 = mu20 / (m00**(1 + (2/2)))
eta02 = mu02 / (m00**(1 + (2/2)))
eta11 = mu11 / (m00**(1 + (2/2)))
eta22 = mu22 / (m00**(1 + (4/2)))

print("\nη20 =", eta20)
print("η02 =", eta02)
print("η11 =", eta11)
print("η22 =", eta22)

# --- 6. Hu Moments ---
mu30 = central_moment(img, 3, 0, x_bar, y_bar)
mu03 = central_moment(img, 0, 3, x_bar, y_bar)
mu12 = central_moment(img, 1, 2, x_bar, y_bar)
mu21 = central_moment(img, 2, 1, x_bar, y_bar)

eta30 = mu30 / (m00**(1 + (3/2)))
eta03 = mu03 / (m00**(1 + (3/2)))
eta12 = mu12 / (m00**(1 + (3/2)))
eta21 = mu21 / (m00**(1 + (3/2)))

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

print("\nHu Moments:")
print("φ1 =", phi1)
print("φ2 =", phi2)
print("φ3 =", phi3)
print("φ4 =", phi4)
print("φ5 =", phi5)
print("φ6 =", phi6)
print("φ7 =", phi7)
