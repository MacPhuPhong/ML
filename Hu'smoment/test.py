import os

def rename_images_in_folder(folder_path, prefix="image", start_index=1, ext="jpg"):
    """
    Đổi tên tất cả ảnh trong folder thành dạng prefix_index.ext

    Args:
        folder_path (str): Đường dẫn folder chứa ảnh.
        prefix (str): Tiền tố tên mới.
        start_index (int): Chỉ số bắt đầu.
        ext (str): Đuôi ảnh muốn lưu (.jpg, .png,...).
    """
    # Lấy danh sách tất cả file trong folder
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg','.png','.jpeg'))]

    # Sắp xếp tên file theo tên hoặc thời gian
    files.sort()

    for i, filename in enumerate(files, start=start_index):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"{prefix}_{i}.{ext}"
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f"Đổi tên: {filename} → {new_filename}")

    print(f"Đã đổi tên {len(files)} ảnh trong '{folder_path}'.")


# ========================
# Ví dụ sử dụng
# ========================
if __name__ == "__main__":
    folder = r"/media/pphong/New Volume/ML & DL/ML/Hu'smoment/dataset_leafs/la_truc"  # Thay đường dẫn folder
    rename_images_in_folder(folder, prefix="La_Truc", start_index=1, ext="jpg")
