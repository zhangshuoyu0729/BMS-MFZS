import torch
from torchvision.transforms import transforms
from PIL import Image
from model import ZeroShotModel
from background_reconstruct import reconstruct_background
import os
import numpy as np
import time
from torch.multiprocessing import set_start_method
import cv2
import glob
import json


try:
    set_start_method('spawn')
except RuntimeError:
    pass

def adjust_features_dim(features, linear_layer):
    adjusted_features = [linear_layer(torch.tensor(f)).squeeze(0).tolist() for f in features if len(f) == 2048]
    return adjusted_features


def update_feature_mapping(new_features, new_labels, feature_mapping_path):
    if os.path.exists(feature_mapping_path):
        feature_mapping = torch.load(feature_mapping_path)
    else:
        feature_mapping = {'features': [], 'labels': []}

    feature_mapping['features'].extend(new_features)
    feature_mapping['labels'].extend(new_labels)

    torch.save(feature_mapping, feature_mapping_path)

def process_region(args):
    model, region, x, y, time_features, spectrum_features, alpha, beta, gamma, feature_mapping_path, linear_layer = args

    linear_layer = torch.nn.Linear(2048, 512)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    region_rgb = region.convert("RGB")
    region_tensor = transform(region_rgb).unsqueeze(0)

    start_time = time.time()
    with torch.no_grad():
        region_features = model.image_feature_extractor(region_tensor)
        S_pred_region = alpha * region_features + beta * time_features + gamma * spectrum_features
        S_pred_region = linear_layer(S_pred_region)

        if os.path.exists(feature_mapping_path):
            feature_mapping = torch.load(feature_mapping_path)
        else:
            feature_mapping = {'features': [], 'labels': []}

        feature_mapping['features'] = adjust_features_dim(feature_mapping['features'], linear_layer)

        if not feature_mapping['features']:
            processing_time = time.time() - start_time
            return x, y, "Feature mapping is empty.", processing_time

        known_features = torch.tensor(feature_mapping['features'])
        similarity_scores = torch.nn.functional.cosine_similarity(S_pred_region, known_features)
        max_similarity, _ = torch.max(similarity_scores, dim=0)

        processing_time = time.time() - start_time
        feature_size = region_features.size()

        return x, y, max_similarity.item(), processing_time, feature_size

def extract_class_names_from_json(json_dir):
    all_classes = set()
    for json_file in glob(os.path.join(json_dir, "*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            labels = data.get("labels", [])
            all_classes.update(labels)
    return sorted(list(all_classes))

def contrast_algorithm(pi_patch):
    img_array = []
    for filename in glob.glob(pi_patch):
        img_re = cv2.imread(filename)

        if img_re is None:
            print(f"Failed to read file: {filename}")  # Debug statement
            continue

        img_gray = cv2.cvtColor(img_re, cv2.COLOR_BGR2GRAY)
        img_array.append(img_gray)


    leng = len(img_array)

    if leng < 3:
        print("Not enough images to process")
        return []

    h, w = img_array[0].shape[:2]
    pad = 9
    diliration = 9
    M = 0

    def patch(a1, b1, img):
        sumg = 0
        i = 0
        for a in range(a1, min(a1 + 3, img.shape[0])):
            for b in range(b1, min(b1 + 3, img.shape[1])):
                sumg += img[a, b]
                i += 1
        aver = sumg / i
        return aver

    def patch1(a1, b1, img, w=5):
        sumg = 0
        i = 0
        for a in range(a1, min(a1 + w, img.shape[0])):
            for b in range(b1, min(b1 + w, img.shape[1])):
                sumg += img[a, b]
                i += 1
        aver = sumg / i
        return aver

    def outside_patch(a1, b1, img, w=9):
        sum = 0
        i = 0
        for abi in range(a1, min(a1 + w, img.shape[0])):
            for aby in range(b1, min(b1 + 2, img.shape[1])):
                sum += int(img[abi, aby])
                i += 1

        for bci in range(b1, min(b1 + w, img.shape[1])):
            for bcx in range(a1, min(a1 + 2, img.shape[0])):
                sum += int(img[bcx, bci])
                i += 1

        for dci in range(a1, min(a1 + w, img.shape[0])):
            for dcy in range(b1, min(b1 + 2, img.shape[1])):
                sum += int(img[dci, dcy])
                i += 1

        for adi in range(b1, min(b1 + w, img.shape[1])):
            for adx in range(a1, min(a1 + 2, img.shape[0])):
                sum += int(img[adx, adi])
                i += 1

        aver = sum / i
        return aver

    recognized_points = []

    for i in range(len(img_array) - 2):
        img1 = img_array[i]
        img2 = img_array[i + 1]
        img3 = img_array[i + 2]

        di = int((diliration - 1) / 2)

        for y in range(0, h - pad + 1):
            for x in range(0, w - pad + 1):
                center_x, center_y = x + di, y + di

                comp_s = abs(
                    (int(img2[center_x, center_y]) - int(patch(x, y, img2))) *
                    (int(img2[2 * center_x - x - 2, 2 * center_y - y - 2]) - int(
                        patch(2 * center_x - x - 2, 2 * center_y - y - 2, img2)))
                )

                comp_t = abs(
                    (int(img1[center_x, center_y]) - int(patch(x, y, img1))) *
                    (int(img1[2 * center_x - x - 2, 2 * center_y - y - 2]) - int(
                        patch(2 * center_x - x - 2, 2 * center_y - y - 2, img1))) -
                    (int(img2[center_x, center_y]) - int(patch(x, y, img2))) *
                    (int(img2[2 * center_x - x - 2, 2 * center_y - y - 2]) - int(
                        patch(2 * center_x - x - 2, 2 * center_y - y - 2, img2))) +
                    (int(img3[center_x, center_y]) - int(patch(x, y, img3))) *
                    (int(img3[2 * center_x - x - 2, 2 * center_y - y - 2]) - int(
                        patch(2 * center_x - x - 2, 2 * center_y - y - 2, img3)))
                )

                inout1 = outside_patch(center_x, center_y, img1, w=9, inw=5) / patch1(center_x, center_y, img1, w=5)
                inout2 = outside_patch(center_x, center_y, img2, w=9, inw=5) / patch1(center_x, center_y, img2, w=5)
                inout3 = outside_patch(center_x, center_y, img3, w=9, inw=5) / patch1(center_x, center_y, img3, w=5)


                comp_c = 0.3 * comp_s + 0.7 * comp_t

                if comp_t != 0:
                    if comp_c > int(0.3 * comp_s) and img2[center_x, center_y] != 0 and patch(center_x, center_y, img2) > patch1(center_x, center_y, img2, w=5):
                        if abs((inout1 / inout2 - inout3 / inout2)) > 0.003:       #uneven  0.0017
                            if [center_x, center_y] not in recognized_points:
                                recognized_points.append([center_x, center_y])

    print(len(recognized_points))

    return recognized_points

def extract_time_features(image_sequence, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    time_features = []
    for image in image_sequence:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model.image_feature_extractor(image_tensor)
            time_features.append(features.squeeze(0).numpy())
    return np.mean(time_features, axis=0)

def test(model, image_folder, image_result_path, image_contrast_path, spectrum_features, save_model_path, feature_mapping_path):
    start_time = time.time()

    # Load trained model weights
    model.load_state_dict(torch.load(save_model_path))
    model.eval()

    # Linear projection layer to adjust feature dimensions
    linear_layer = torch.nn.Linear(2048, 512)
    linear_layer.eval()

    # Image pre-processing transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load all image files
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    for i in range(1, len(image_files) - 1):
        # Load three consecutive frames for temporal feature extraction
        img_sequence = [
            cv2.imread(image_files[i - 1]),
            cv2.imread(image_files[i]),
            cv2.imread(image_files[i + 1])
        ]

        if any(img is None for img in img_sequence):
            print(f"Failed to read sequence around: {image_files[i]}")
            continue

        # Extract temporal features
        time_features = extract_time_features(img_sequence, model)
        time_features = torch.tensor(time_features).unsqueeze(0)

        current_image_path = image_files[i]
        current_image = Image.open(current_image_path).convert("RGB")
        image_tensor = transform(current_image).unsqueeze(0)

        # Handle spectral features (if missing, initialize to zeros)
        if spectrum_features is None:
            spectrum_features = torch.zeros_like(image_tensor)

        # --- Background Reconstruction ---
        background_path = os.path.join(image_result_path, "background_" + os.path.basename(current_image_path))
        reconstruct_background(current_image_path, model, feature_mapping_path, background_path)
        reconstructed_background = Image.open(background_path).convert("L")
        reconstructed_array = np.array(reconstructed_background)

        # Extract current image features
        with torch.no_grad():
            image_features = model.image_feature_extractor(image_tensor)

        # Weighted fusion of spatial, temporal, and spectral features
        alpha, beta, gamma = 0.6, 0.2, 0.2
        S_pred = alpha * image_features + beta * time_features + gamma * spectrum_features
        S_pred = linear_layer(S_pred)

        # Load grayscale version of current image
        gray_image = Image.open(current_image_path).convert("L")
        image_width, image_height = gray_image.size
        img_array = np.array(gray_image)

        # --- Background subtraction using reconstruction ---
        img_array = img_array - reconstructed_array
        img_array = np.clip(img_array, 0, 255)

        window_size = 5
        stride = 2

        # Divide image into sliding windows
        regions = []
        for y in range(0, image_height - window_size + 1, stride):
            for x in range(0, image_width - window_size + 1, stride):
                region = gray_image.crop((x, y, x + window_size, y + window_size))
                regions.append((model, region, x, y, time_features, spectrum_features, alpha, beta, gamma,
                                feature_mapping_path, linear_layer))

        # Perform region-level feature matching
        results = [process_region(args) for args in regions]

        # Collect high-confidence regions based on similarity score
        recognized_points_fm = []
        for x, y, result, processing_time, feature_size in results:
            if isinstance(result, str):
                continue
            elif result >= 0.598:  # Similarity threshold
                recognized_points_fm.append((x, y))

        print(f"[{os.path.basename(current_image_path)}] Feature-matched regions: {len(recognized_points_fm)}")

        # Apply contrast-based algorithm
        contrast_points = contrast_algorithm(image_contrast_path)

        # Retain only consensus regions between both methods
        recognized_points_contrast = [pt for pt in contrast_points if pt in recognized_points_fm]

        print(f"[{os.path.basename(current_image_path)}] Final target regions after dual verification: {len(recognized_points_contrast)}")

        # Suppress background by dimming non-target regions
        for y in range(0, image_height - window_size + 1, stride):
            for x in range(0, image_width - window_size + 1, stride):
                if (x, y) not in recognized_points_contrast:
                    avg_val = np.mean(img_array[y:y + window_size, x:x + window_size])
                    img_array[y:y + window_size, x:x + window_size] = avg_val * 0.1

        # Highlight detected target regions
        for (cx, cy) in recognized_points_contrast:
            if 0 <= cx < img_array.shape[1] and 0 <= cy < img_array.shape[0]:
                img_array[cy:cy + window_size, cx:cx + window_size] = 200
            else:
                print(f"Point ({cx}, {cy}) out of bounds")

        # Save final result
        new_background_features = []
        new_background_labels = []

        for y in range(0, image_height - window_size + 1, stride):
            for x in range(0, image_width - window_size + 1, stride):
                if (x, y) not in recognized_points_contrast:
                    region = gray_image.crop((x, y, x + window_size, y + window_size)).convert("RGB")
                    region_tensor = transform(region).unsqueeze(0)
                    with torch.no_grad():
                        region_feat = model.image_feature_extractor(region_tensor)
                        region_feat_proj = linear_layer(region_feat).squeeze(0).cpu().numpy().tolist()
                        new_background_features.append(region_feat_proj)
                        new_background_labels.append([0] * model.num_classes)

        if new_background_features:
            update_feature_mapping(new_background_features, new_background_labels, feature_mapping_path)
            print(f"Added {len(new_background_features)} new background features to mapping.")

        # Save final result
        new_background_features = []
        new_background_labels = []

        for y in range(0, image_height - window_size + 1, stride):
            for x in range(0, image_width - window_size + 1, stride):
                if (x, y) not in recognized_points_contrast:
                    region = gray_image.crop((x, y, x + window_size, y + window_size)).convert("RGB")
                    region_tensor = transform(region).unsqueeze(0)
                    with torch.no_grad():
                        region_feat = model.image_feature_extractor(region_tensor)
                        region_feat_proj = linear_layer(region_feat).squeeze(0).cpu().numpy().tolist()
                        new_background_features.append(region_feat_proj)
                        new_background_labels.append([0] * model.num_classes)

        if new_background_features:
            update_feature_mapping(new_background_features, new_background_labels, feature_mapping_path)
            print(f"Added {len(new_background_features)} new background features to mapping.")

        # Save final result
        modified_image = Image.fromarray(np.uint8(img_array))
        save_path = os.path.join(image_result_path, "modified_" + os.path.basename(current_image_path))
        modified_image.save(save_path)

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")



if __name__ == '__main__':
    save_model_path = r"./starry-data/model1/best_model.pth"
    feature_mapping_path = r"./starry-data/feature_mapping_1.pkl"
    spectrum_features = r"./starry-data/spec.csv"
    train_json_dir = "./starry-data/label-database/"
    image_folder = r"./starry-data/data3"
    image_contrast_path = r"./starry-data/data3*.jpg"
    image_result_path = "./starry-data/result/3"


    class_names = extract_class_names_from_json(train_json_dir)

    model = ZeroShotModel(
        image_feature_dim=2048,
        text_feature_dim=512,
        num_classes= len(class_names),
        time_feature_dim=2050,
        spectrum_feature_dim=512,
        spectrum_hidden_dim=1024,
        spectrum_num_layers=4,
        spectrum_num_heads=8,
        spectrum_dropout=0.1
    )

    # Reconstruct the background of the current image
    background_path = os.path.join(image_result_path, "background_" + os.path.basename(image_folder))
    reconstruct_background(image_folder, model, feature_mapping_path, background_path)

    # Load the reconstructed image for fusion processing
    reconstructed_background = Image.open(background_path).convert("L")
    reconstructed_array = np.array(reconstructed_background)

    test(model, image_folder, image_result_path, image_contrast_path, spectrum_features, save_model_path, feature_mapping_path)
