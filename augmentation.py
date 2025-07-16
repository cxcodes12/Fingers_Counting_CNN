import os
import cv2
import random
import numpy as np

def rotate_image_cv(img, angle_deg):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def adjust_contrast_cv(img, factor):
    img_float = img.astype(np.float32)
    mean = np.mean(img_float)
    img_contrasted = (img_float - mean) * factor + mean
    img_contrasted = np.clip(img_contrasted, 0, 255).astype(np.uint8)
    return img_contrasted

def add_gaussian_noise(img, mean=0, stddev=10):
    noise = np.random.normal(mean, stddev, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def augment_dataset(src_folder, dst_folder):
    for root, dirs, files in os.walk(src_folder):
        rel_path = os.path.relpath(root, src_folder)
        target_dir = os.path.join(dst_folder, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        image_files.sort()

        for idx, file in enumerate(image_files):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_dir, file)

            img = cv2.imread(src_path)
            if img is None:
                print(f"Eroare la citirea imaginii: {src_path}")
                continue

        
            cv2.imwrite(dst_path, img)

            # one out of every two images
            if (idx + 1) % 2 == 0:
                base_name, ext = os.path.splitext(file)

                # rotation
                angle = random.uniform(-20, 20)
                img_rotated = rotate_image_cv(img, angle)

                # contrast adjustment 
                factor = random.uniform(0.7, 1.3)
                img_contrasted = adjust_contrast_cv(img_rotated, factor)

                # gaussian noise
                stddev = random.uniform(5, 20)  # random deviation
                img_augmented = add_gaussian_noise(img_contrasted, stddev=stddev)

                # save
                aug_path = os.path.join(target_dir, f"{base_name}_augmented{ext}")
                cv2.imwrite(aug_path, img_augmented)

    print(f"Augmentation done! images saved in {dst_folder}")


src_folder = r"path source folder"
dst_folder = r"path destination folder for the augmented training set"

augment_dataset(src_folder, dst_folder)