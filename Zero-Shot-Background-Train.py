import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms
from model import ZeroShotModel
from data_gen import CustomDataset
from text_feature_extraction import BertTokenizer  # 导入tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pandas as pd
import glob
import json

def tsne_target_background_overlay_center_jitter(
        feature_mapping_path,
        class_csv_path,
        save_path,
        alpha=0.6,
        point_size=15,
        color_fade=0.8,
):
    # Step 1: Loading feature maps
    mapping = torch.load(feature_mapping_path)
    features = np.array(mapping["features"])
    labels = np.array(mapping["labels"])
    class_names = mapping["class_names"]

    # Step 2: Read target and background categories from CSV
    class_df = pd.read_csv(class_csv_path)
    target_classes = class_df[class_df['type'] == 'target']['class'].tolist()
    background_classes = class_df[class_df['type'] == 'background']['class'].tolist()

    # Step 3: Constructing binary labels
    target_idx = [class_names.index(cls) for cls in target_classes if cls in class_names]
    background_idx = [class_names.index(cls) for cls in background_classes if cls in class_names]

    binary_labels = []
    for label_vec in labels:
        is_target = any(label_vec[i] > 0.5 for i in target_idx)
        is_background = any(label_vec[i] > 0.5 for i in background_idx)
        if is_target:
            binary_labels.append(1)
        elif is_background:
            binary_labels.append(0)
        else:
            binary_labels.append(-1)
    binary_labels = np.array(binary_labels)

    # Step 4: Claculate t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_result = tsne.fit_transform(features)
    tsne_x = tsne_result[:, 0]
    tsne_y = tsne_result[:, 1]

    # Step 5: Visualization
    plt.figure(figsize=(10, 8))

    light_red = (1.0, color_fade * 0.8, color_fade * 0.8)
    light_blue = (color_fade * 0.8, color_fade * 0.8, 1.0)
    light_gray = (0.8, 0.8, 0.8)

    plt.scatter(
        tsne_x[binary_labels == 1],
        tsne_y[binary_labels == 1],
        c=[light_red],
        label='Target',
        alpha=alpha,
        s=point_size,
        edgecolors=[light_red]
    )
    plt.scatter(
        tsne_x[binary_labels == 0],
        tsne_y[binary_labels == 0],
        c=[light_blue],
        label='Background',
        alpha=alpha,
        s=point_size,
        edgecolors=[light_blue]
    )
    plt.scatter(
        tsne_x[binary_labels == -1],
        tsne_y[binary_labels == -1],
        c=[light_gray],
        label='Other',
        alpha=alpha * 0.7,
        s=point_size * 0.8
    )

    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend()
    plt.tight_layout()

    save_fp = os.path.join(save_path, f"tsne_overlay.png")
    plt.savefig(save_fp, dpi=300)
    plt.close()

def inspect_feature_mapping(feature_mapping_path, class_names):
    mapping = torch.load(feature_mapping_path)
    labels = np.array(mapping["labels"])

    label_sums = labels.sum(axis=1)

def extract_class_names_from_json(json_dir):
    all_classes = set()
    for json_file in glob(os.path.join(json_dir, "*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            labels = data.get("labels", [])
            all_classes.update(labels)
    return sorted(list(all_classes))

def update_feature_mapping(new_features, new_labels, feature_mapping_path):
    if os.path.exists(feature_mapping_path):
        feature_mapping = torch.load(feature_mapping_path)
    else:
        feature_mapping = {'features': [], 'labels': []}

    feature_mapping['features'].extend(new_features)
    feature_mapping['labels'].extend(new_labels)
    torch.save(feature_mapping, feature_mapping_path)

def similarity_score(image_feat, label_feat, time_feat, spectrum_feat, alpha=1.0, beta=1.0, gamma=1.0):
    # Normalize vectors
    def cosine_sim(a, b):
        return torch.sum(a * b, dim=1) / (torch.norm(a, dim=1) * torch.norm(b, dim=1) + 1e-8)

    sim_il = cosine_sim(image_feat, label_feat)
    sim_it = cosine_sim(image_feat, time_feat)
    sim_is = cosine_sim(image_feat, spectrum_feat)

    return alpha * sim_il + beta * sim_it + gamma * sim_is

def similarity_aware_loss(predicted, actual, alpha=0.2, beta=0.4, gamma=0.4):
    i_feat = predicted["image"]
    l_feat = actual["text"]
    t_feat = actual["temporal"]
    s_feat = actual["spectral"]

    sim_pred = similarity_score(i_feat, l_feat, t_feat, s_feat, alpha, beta, gamma)
    sim_actual = similarity_score(i_feat, l_feat, t_feat, s_feat, alpha, beta, gamma)

    loss = -torch.mean(sim_actual * torch.log(sim_pred + 1e-8))
    return loss, loss.item()

def custom_collate_fn(batch):
    """
    接收一个 batch 的 [(image, label, time_feat, spec_feat, path), ...]
    返回 batched tensors
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    time_features = torch.stack([item[2] for item in batch])
    spectrum_features = torch.stack([item[3] for item in batch])
    image_paths = [item[4] for item in batch]

    return images, labels, time_features, spectrum_features, image_paths

def train(model, train_loader, val_loader, optimizer, num_epochs, save_model_path, feature_mapping_path, class_names, tokenizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.train()

    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    train_loss_list, val_loss_list = [], []
    f1_list = []
    val_f1_list = []

    for epoch in range(num_epochs):
        model.train()
        running_total_loss = 0.0

        y_true_all, y_pred_all = [], []
        epoch_features, epoch_labels = [], []
        all_sigmoid_outputs = []

        for batch in train_loader:
            if batch is None:
                continue

            images, labels, time_features, spectrum_features, image_paths = batch
            images = images.to(device)
            labels = labels.to(device)
            time_features = time_features.to(device)
            spectrum_features = spectrum_features.to(device)

            optimizer.zero_grad()

            # Convert labels to text
            batch_label_texts = []
            for label_vec in labels:
                idxs = (label_vec > 0).nonzero(as_tuple=True)[0]
                text_label = " ".join([class_names[idx.item()] for idx in idxs])
                batch_label_texts.append(text_label)

            tokenized = tokenizer(batch_label_texts, return_tensors='pt', padding=True, truncation=True)
            text_inputs = {k: v.to(device) for k, v in tokenized.items()}

            outputs1, outputs, text_features, outputs_intermediate = model(images, text_inputs, time_features, spectrum_features)

            # Average pool
            pooled_time = time_features.mean(dim=1) if time_features.dim() == 3 else time_features
            pooled_spectrum = spectrum_features.mean(dim=1) if spectrum_features.dim() == 3 else spectrum_features

            actual_similarity = {
                "text": text_features,
                "temporal": pooled_time,
                "spectral": pooled_spectrum
            }
            predicted_similarity = {
                "image": outputs_intermediate,
                "temporal": pooled_time,
                "spectral": pooled_spectrum
            }

            similarity_loss, _ = similarity_aware_loss(predicted_similarity, actual_similarity)
            bce_loss = bce_loss_fn(outputs1, labels.float())
            total_loss = similarity_loss + 0.5 * bce_loss

            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item()

            epoch_features.extend(text_features.detach().cpu().numpy().tolist())
            epoch_labels.extend(labels.detach().cpu().numpy().tolist())

            sigmoid_outputs = torch.sigmoid(outputs1.detach().cpu())
            all_sigmoid_outputs.append(sigmoid_outputs)
            y_true_all.append(labels.detach().cpu())
            y_pred_all.append((sigmoid_outputs > 0.5).int())

        y_true_all = torch.cat(y_true_all).int().numpy()
        y_pred_all = torch.cat(y_pred_all).int().numpy()
        all_sigmoid_outputs = torch.cat(all_sigmoid_outputs).numpy()

        macro_f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true_all, y_pred_all, average='micro', zero_division=0)

        train_loss = running_total_loss / len(train_loader)
        train_loss_list.append(train_loss)
        f1_list.append(macro_f1)

        print(f"[Train] Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Macro-F1: {macro_f1:.4f}, Micro-F1: {micro_f1:.4f}")

        # 输出分布可视化
        plt.figure()
        for i in range(all_sigmoid_outputs.shape[1]):
            plt.hist(all_sigmoid_outputs[:, i], bins=20, alpha=0.5, label=f'class {class_names[i]}')
        plt.title(f"Sigmoid Output Distribution - Epoch {epoch+1}")
        plt.xlabel("Sigmoid Value")
        plt.ylabel("Frequency")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_model_path, f"output_distribution_epoch_{epoch+1}.png"))
        plt.close()

        # Validation
        model.eval()
        val_loss_total = 0
        val_y_true, val_y_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, labels, time_features, spectrum_features, _ = batch
                images = images.to(device)
                labels = labels.to(device)
                time_features = time_features.to(device)
                spectrum_features = spectrum_features.to(device)

                batch_label_texts = []
                for label_vec in labels:
                    idxs = (label_vec > 0).nonzero(as_tuple=True)[0]
                    text_label = " ".join([class_names[idx.item()] for idx in idxs])
                    batch_label_texts.append(text_label)

                tokenized = tokenizer(batch_label_texts, return_tensors='pt', padding=True, truncation=True)
                text_inputs = {k: v.to(device) for k, v in tokenized.items()}

                outputs1, outputs, text_features, outputs_intermediate = model(images, text_inputs, time_features, spectrum_features)

                pooled_time = time_features.mean(dim=1) if time_features.dim() == 3 else time_features
                pooled_spectrum = spectrum_features.mean(dim=1) if spectrum_features.dim() == 3 else spectrum_features

                actual_similarity = {
                    "text": text_features,
                    "temporal": pooled_time,
                    "spectral": pooled_spectrum
                }
                predicted_similarity = {
                    "image": outputs_intermediate,
                    "temporal": pooled_time,
                    "spectral": pooled_spectrum
                }

                val_loss, _ = similarity_aware_loss(predicted_similarity, actual_similarity)
                bce_val_loss = bce_loss_fn(outputs1, labels.float())
                total_val_loss = val_loss + 0.5 * bce_val_loss
                val_loss_total += total_val_loss.item()

                val_y_true.append(labels.cpu())
                val_y_pred.append((torch.sigmoid(outputs1.cpu()) > 0.5).int())

        val_y_true = torch.cat(val_y_true).numpy()
        val_y_pred = torch.cat(val_y_pred).numpy()

        val_macro_f1 = f1_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_micro_f1 = f1_score(val_y_true, val_y_pred, average='micro', zero_division=0)
        val_loss_avg = val_loss_total / len(val_loader)

        val_loss_list.append(val_loss_avg)
        val_f1_list.append(val_macro_f1)

        print(f"[Validation] Loss: {val_loss_avg:.4f}, Macro-F1: {val_macro_f1:.4f}, Micro-F1: {val_micro_f1:.4f}")

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            torch.save(model.state_dict(), os.path.join(save_model_path, "best_model.pth"))

        scheduler.step()
        update_feature_mapping(epoch_features, epoch_labels, feature_mapping_path)

    torch.save(model.state_dict(), os.path.join(save_model_path, "final_model.pth"))

    plt.figure()
    plt.plot(train_loss_list, label="Train Loss", marker='o')
    plt.plot(val_loss_list, label="Val Loss", marker='x')
    plt.plot(f1_list, label="Train Macro-F1", marker='s')
    plt.plot(val_f1_list, label="Val Macro-F1", marker='d')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training & Validation Loss and F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_model_path, "train_val_loss_f1.png"))
    plt.close()

    print("The model training is complete and all results have been saved.")


if __name__ == "__main__":
    learning_rate = 0.001
    num_epochs = 300
    batch_size = 32

    train_data_dir_1 = "./starry-data/1/"
    train_json_dir_1 = "./starry-data/1-lable/"
    train_data_dir_2 = "./starry-data/2/"
    train_json_dir_2 = "./starry-data/2-lable/"

    train_spectral_dir1 = "./starry-data/spectral1/"
    train_spectral_dir2 = "./starry-data/spectral2/"

    save_model_path = "E:/starry-data/model1/"
    feature_mapping_path = "E:/starry-data/feature_mapping_1.pkl"


    os.makedirs(save_model_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    class_names = extract_class_names_from_json(train_json_dir_1)

    # Read the original dataset
    dataset_1 = CustomDataset(
        image_dir=train_data_dir_1,
        json_dir=train_json_dir_1,
        spectral_dir=train_spectral_dir1,
        transform=transform
    )

    dataset_2 = CustomDataset(
        image_dir=train_data_dir_2,
        json_dir=train_json_dir_2,
        spectral_dir=train_spectral_dir2,
        transform=transform
    )

    full_dataset = ConcatDataset([dataset_1, dataset_2])

    total_indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(total_indices, test_size=0.2, random_state=42, shuffle=True)

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize and optimizer
    tokenizer = BertTokenizer()

    model = ZeroShotModel(
        image_feature_dim=2048,
        text_feature_dim=512,
        num_classes=len(class_names),
        time_feature_dim=2050,
        spectrum_feature_dim=512,
        spectrum_hidden_dim=1024,
        spectrum_num_layers=4,
        spectrum_num_heads=8,
        spectrum_dropout=0.1
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        save_model_path=save_model_path,
        feature_mapping_path=feature_mapping_path,
        class_names=class_names,
        tokenizer=tokenizer,
        scheduler=scheduler
    )

    # Post-processing and visualization
    inspect_feature_mapping(feature_mapping_path, class_names)
    tsne_target_background_overlay_center_jitter(
        feature_mapping_path,
        class_names,
        save_model_path,
    )