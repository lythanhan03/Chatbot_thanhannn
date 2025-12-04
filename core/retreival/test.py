import numpy as np
import faiss

# Tạo dữ liệu giả lập
d = 64  # chiều của vector
nb = 10000  # số lượng vector trong tập dữ liệu
nq = 10  # số lượng vector truy vấn
np.random.seed(123)
xb = np.random.random((nb, d)).astype('float32')  # tập dữ liệu
xq = np.random.random((nq, d)).astype('float32')  # truy vấn

# Tạo index với Exact Search (L2 distance)
index = faiss.IndexFlatL2(d)
index.add(xb)  # Thêm dữ liệu vào index

# Tìm kiếm 5 neighbor gần nhất
k = 5
distances, indices = index.search(xq, k)

print("Khoảng cách:", distances)
print("Chỉ số:", indices)