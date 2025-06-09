import os
import glob
import json
import numpy as np
from PIL import Image
from scipy.signal import spectrogram
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, f1_score, precision_score, \
    recall_score

# ─── 1. 설정 ─────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# TODO: 자신의 실제 데이터 경로로 수정하세요
RF_BASE_DIR = r'C:\Users\이지은\Desktop\Python\pythonProject2\dataset'
IMG_BASE_DIR = r'C:\Users\이지은\Desktop\Python\pythonProject2\dataset\MVTec'


BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 30
LATENT_DIM = 512
SEQUENCE_LENGTH = 10  # LSTM을 위한 시퀀스 길이


# ─── 2. Dataset 정의 ─────────────────────────────────────────────────────
class RFSpectrogramDataset(Dataset):
    """MIT RF Dataset: IQ → Spectrogram → Temporal Sequences"""

    def __init__(self, base_dir, transform=None, train=True, sequence_length=SEQUENCE_LENGTH):
        self.files = glob.glob(os.path.join(base_dir, '**', '*.sigmf-meta'), recursive=True)
        self.transform = transform
        self.train = train
        self.sequence_length = sequence_length

        if len(self.files) == 0:
            raise ValueError(f"No RF data files found in {base_dir}")

        print(f"Found {len(self.files)} RF files")

        # Train/Test split (80/20)
        split_idx = int(len(self.files) * 0.8)
        if train:
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

    def __len__(self):
        return len(self.files)

    def _create_temporal_sequence(self, spec):
        """스펙트로그램을 시간축으로 분할하여 시퀀스 생성"""
        _, time_steps = spec.shape

        if time_steps < self.sequence_length:
            # 패딩
            pad_width = self.sequence_length - time_steps
            spec = np.pad(spec, ((0, 0), (0, pad_width)), 'constant')
            time_steps = self.sequence_length

        # 시간축을 sequence_length 단위로 분할
        step_size = time_steps // self.sequence_length
        sequences = []

        for i in range(self.sequence_length):
            start_idx = i * step_size
            end_idx = start_idx + step_size if i < self.sequence_length - 1 else time_steps
            time_slice = spec[:, start_idx:end_idx]

            # 각 time slice를 224x224로 리사이즈
            if time_slice.shape[1] < 224:
                time_slice = np.pad(time_slice, ((0, max(0, 224 - time_slice.shape[0])),
                                                 (0, 224 - time_slice.shape[1])), 'constant')

            time_slice = time_slice[:224, :224]
            sequences.append(time_slice)

        return np.array(sequences)  # [sequence_length, freq, time]

    def __getitem__(self, idx):
        try:
            meta_path = self.files[idx]
            data_path = meta_path.replace('.sigmf-meta', '.sigmf-data')

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            # IQ 데이터 로드
            iq = np.fromfile(data_path, dtype=np.complex64)
            if len(iq) == 0:
                raise ValueError(f"Empty data file: {data_path}")

            # 메타데이터 로드
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                sr = metadata.get('global', {}).get('core:sample_rate', 1e6)

            # 스펙트로그램 생성 (더 긴 시간 윈도우)
            if len(iq) < 2048:
                iq = np.pad(iq, (0, 2048 - len(iq)), 'constant')

            f, t, Sxx = spectrogram(iq, fs=sr, window='hann',
                                    nperseg=512, noverlap=256)  # 더 많은 시간 정보

            # 로그 스케일 변환
            spec = 10 * np.log10(np.abs(Sxx) + 1e-10)

            # 정규화
            spec_min, spec_max = spec.min(), spec.max()
            if spec_max > spec_min:
                spec = (spec - spec_min) / (spec_max - spec_min)
            else:
                spec = np.zeros_like(spec)

            # 시간적 시퀀스 생성
            temporal_seq = self._create_temporal_sequence(spec.astype(np.float32))

            # 3채널로 변환
            temporal_seq_3ch = np.stack([temporal_seq, temporal_seq, temporal_seq], axis=1)
            x = torch.from_numpy(temporal_seq_3ch)  # [seq_len, 3, freq, time]

            if self.transform:
                # 각 시퀀스에 transform 적용
                transformed_seq = []
                for i in range(x.size(0)):
                    transformed_seq.append(self.transform(x[i]))
                x = torch.stack(transformed_seq)

            return x, 0  # [seq_len, 3, 224, 224]

        except Exception as e:
            print(f"Error processing {meta_path}: {e}")
            # 더미 시퀀스 반환
            dummy = torch.zeros(self.sequence_length, 3, 224, 224)
            if self.transform:
                transformed_seq = []
                for i in range(dummy.size(0)):
                    transformed_seq.append(self.transform(dummy[i]))
                dummy = torch.stack(transformed_seq)
            return dummy, 0


class MVTecDataset(Dataset):
    """MVTec AD Dataset"""

    def __init__(self, base_dir, categories=None, train=True, transform=None):
        self.paths = []
        self.labels = []

        if not os.path.exists(base_dir):
            raise ValueError(f"Base directory does not exist: {base_dir}")

        if categories is None:
            categories = [d for d in os.listdir(base_dir)
                          if os.path.isdir(os.path.join(base_dir, d))]

        if not categories:
            raise ValueError(f"No categories found in {base_dir}")

        mode = 'train' if train else 'test'

        for cat in categories:
            mode_dir = os.path.join(base_dir, cat, mode)
            if not os.path.isdir(mode_dir):
                print(f"Warning: {mode_dir} not found, skipping category {cat}")
                continue

            if train:
                good_dir = os.path.join(mode_dir, 'good')
                if os.path.isdir(good_dir):
                    for fn in os.listdir(good_dir):
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            self.paths.append(os.path.join(good_dir, fn))
                            self.labels.append(0)
            else:
                for cls in os.listdir(mode_dir):
                    cls_dir = os.path.join(mode_dir, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    for fn in os.listdir(cls_dir):
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            self.paths.append(os.path.join(cls_dir, fn))
                            self.labels.append(0 if cls == 'good' else 1)

        self.transform = transform

        if len(self.paths) == 0:
            raise ValueError(f"No valid images found for {mode} mode")

        print(f"MVTec {mode}: {len(self.paths)} images, "
              f"Normal: {sum(1 for l in self.labels if l == 0)}, "
              f"Anomaly: {sum(1 for l in self.labels if l == 1)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {self.paths[idx]}: {e}")
            dummy = torch.zeros(3, 224, 224)
            if self.transform:
                dummy = self.transform(dummy)
            return dummy, self.labels[idx]


# ─── 3. Transform 정의 ─────────────────────────────────────────────────────
rf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── 4. Feature Extraction (ResNet50) ─────────────────────────────────────
class FeatureExtractor(nn.Module):
    """다이어그램의 Feature Extraction 모듈"""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # ResNet50 backbone
        backbone = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # GAP 전까지
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(backbone.fc.in_features, latent_dim)

    def forward(self, x):
        # x shape: [batch, 3, 224, 224]
        features = self.features(x)  # [batch, 2048, 7, 7]
        spatial_features = self.spatial_pool(features).view(features.size(0), -1)  # [batch, 2048]
        spatial_feature_maps = self.fc(spatial_features)  # [batch, latent_dim]
        return spatial_feature_maps


# ─── 5. Temporal Analysis (LSTM) ──────────────────────────────────────────
class TemporalAnalysis(nn.Module):
    """다이어그램의 Temporal Analysis 모듈"""

    def __init__(self, input_dim=LATENT_DIM, hidden_dim=LATENT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.temporal_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # bidirectional

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch, seq_len, hidden_dim*2]
        # 마지막 타임스텝의 출력 사용
        temporal_dependencies = self.temporal_fc(lstm_out[:, -1, :])  # [batch, hidden_dim]
        return temporal_dependencies


# ─── 6. 전체 모델 (다이어그램 구조 반영) ─────────────────────────────────────
class MultiModalArchitecture(nn.Module):
    """다이어그램 구조를 정확히 반영한 모델"""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # Feature Extraction
        self.rf_feature_extractor = FeatureExtractor(latent_dim)
        self.img_feature_extractor = FeatureExtractor(latent_dim)

        # Temporal Analysis (RF 신호용)
        self.temporal_analysis = TemporalAnalysis(latent_dim, latent_dim)

        # Model Output (Anomaly Detection)
        self.anomaly_detector = nn.Sequential(
            nn.Linear(latent_dim * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # 재구성을 위한 디코더 (unsupervised learning)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(2048, 3 * 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, rf_sequence, img_single):
        # RF 신호 처리 (시간적 시퀀스)
        batch_size, seq_len = rf_sequence.size(0), rf_sequence.size(1)

        # 각 시간 프레임에서 spatial feature 추출
        rf_features = []
        for t in range(seq_len):
            rf_frame = rf_sequence[:, t]  # [batch, 3, 224, 224]
            rf_feat = self.rf_feature_extractor(rf_frame)  # [batch, latent_dim]
            rf_features.append(rf_feat)

        rf_feature_sequence = torch.stack(rf_features, dim=1)  # [batch, seq_len, latent_dim]

        # Temporal Analysis
        temporal_dependencies = self.temporal_analysis(rf_feature_sequence)  # [batch, latent_dim]

        # 이미지 feature 추출
        img_spatial_features = self.img_feature_extractor(img_single)  # [batch, latent_dim]

        # Feature 융합
        fused_features = torch.cat([temporal_dependencies, img_spatial_features], dim=1)

        # Anomaly Detection Output
        anomaly_score = self.anomaly_detector(fused_features)

        # 재구성 (unsupervised learning용)
        reconstruction = self.decoder(fused_features).view(-1, 3, 224, 224)

        return {
            'anomaly_score': anomaly_score,
            'reconstruction': reconstruction,
            'rf_features': temporal_dependencies,
            'img_features': img_spatial_features,
            'fused_features': fused_features
        }


# ─── 7. 커스텀 데이터로더 ──────────────────────────────────────────────────
class SyncedDataLoader:
    def __init__(self, rf_loader, img_loader):
        self.rf_loader = rf_loader
        self.img_loader = img_loader
        self.rf_iter = iter(rf_loader)
        self.img_iter = iter(img_loader)

    def __iter__(self):
        self.rf_iter = iter(self.rf_loader)
        self.img_iter = iter(self.img_loader)
        return self

    def __next__(self):
        try:
            rf_batch = next(self.rf_iter)
        except StopIteration:
            self.rf_iter = iter(self.rf_loader)
            rf_batch = next(self.rf_iter)

        try:
            img_batch = next(self.img_iter)
        except StopIteration:
            self.img_iter = iter(self.img_loader)
            img_batch = next(self.img_iter)

        return rf_batch, img_batch

    def __len__(self):
        return max(len(self.rf_loader), len(self.img_loader))


# ─── 8. 학습/평가 함수 ─────────────────────────────────────────────────────
def train_epoch(model, train_loader, optimizer, criterion_recon, criterion_anomaly):
    model.train()
    total_loss = 0
    num_batches = 0

    for (rf_seq, _), (img_x, img_labels) in train_loader:
        rf_seq, img_x = rf_seq.to(DEVICE), img_x.to(DEVICE)
        img_labels = img_labels.to(DEVICE).float()

        # 배치 크기 맞추기
        min_batch = min(rf_seq.size(0), img_x.size(0))
        rf_seq, img_x = rf_seq[:min_batch], img_x[:min_batch]
        img_labels = img_labels[:min_batch]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(rf_seq, img_x)

        # Loss 계산
        recon_loss = criterion_recon(outputs['reconstruction'], img_x)

        # Unsupervised 학습이므로 anomaly loss는 reconstruction error 기반
        anomaly_loss = criterion_recon(outputs['reconstruction'], img_x)

        total_loss_batch = recon_loss + 0.1 * anomaly_loss
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()
        num_batches += 1

        if num_batches >= 100:
            break

    return total_loss / num_batches if num_batches > 0 else 0


@torch.no_grad()
def eval_epoch(model, test_loader, criterion_recon):
    model.eval()
    total_loss = 0
    num_batches = 0

    for (rf_seq, _), (img_x, _) in test_loader:
        rf_seq, img_x = rf_seq.to(DEVICE), img_x.to(DEVICE)

        min_batch = min(rf_seq.size(0), img_x.size(0))
        rf_seq, img_x = rf_seq[:min_batch], img_x[:min_batch]

        outputs = model(rf_seq, img_x)
        loss = criterion_recon(outputs['reconstruction'], img_x)
        total_loss += loss.item()
        num_batches += 1

        if num_batches >= 50:
            break

    return total_loss / num_batches if num_batches > 0 else float('inf')


@torch.no_grad()
def compute_performance_metrics(model, test_loader):
    """Model Optimization 단계의 Performance Metrics 계산"""
    model.eval()
    scores, labels = [], []

    for (rf_seq, _), (img_x, lbl) in test_loader:
        rf_seq, img_x = rf_seq.to(DEVICE), img_x.to(DEVICE)

        min_batch = min(rf_seq.size(0), img_x.size(0))
        rf_seq, img_x = rf_seq[:min_batch], img_x[:min_batch]
        lbl = lbl[:min_batch]

        outputs = model(rf_seq, img_x)

        # Anomaly score는 reconstruction error 사용
        recon_error = ((outputs['reconstruction'] - img_x) ** 2).view(rf_seq.size(0), -1).mean(dim=1)

        scores.extend(recon_error.cpu().numpy())
        labels.extend(lbl.cpu().numpy())

    scores = np.array(scores)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in test data")
        return {}

    # Performance Metrics 계산
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # Threshold 결정
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Binary predictions
    binary_pred = (scores > optimal_threshold).astype(int)
    accuracy = (binary_pred == labels).mean()
    f1 = f1_score(labels, binary_pred)
    precision_score_val = precision_score(labels, binary_pred)
    recall_score_val = recall_score(labels, binary_pred)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision_score_val,
        'recall': recall_score_val,
        'threshold': optimal_threshold
    }


# ─── 9. 메인 함수 ─────────────────────────────────────────────────────────
def main():
    try:
        print("=== 다이어그램 구조 기반 멀티모달 이상 탐지 시스템 ===")

        # 1. Input Data 단계
        print("\n1. Input Data Processing...")

        if not os.path.exists(IMG_BASE_DIR):
            print(f"Warning: Image directory {IMG_BASE_DIR} not found")
            return

        categories = [d for d in os.listdir(IMG_BASE_DIR)
                      if os.path.isdir(os.path.join(IMG_BASE_DIR, d))]

        if not categories:
            print("No categories found in MVTec directory")
            return

        print(f"Found categories: {categories}")

        # 2. Data Processing 단계
        print("\n2. Data Processing...")
        print("   - IQ to Spectrogram conversion")
        print("   - Image Normalization")

        rf_train_dataset = RFSpectrogramDataset(RF_BASE_DIR, transform=rf_transform, train=True)
        rf_test_dataset = RFSpectrogramDataset(RF_BASE_DIR, transform=rf_transform, train=False)

        mvtec_train_dataset = MVTecDataset(IMG_BASE_DIR, categories, train=True, transform=img_transform)
        mvtec_test_dataset = MVTecDataset(IMG_BASE_DIR, categories, train=False, transform=img_transform)

        # DataLoader 생성
        rf_train_loader = DataLoader(rf_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        rf_test_loader = DataLoader(rf_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        mvtec_train_loader = DataLoader(mvtec_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        mvtec_test_loader = DataLoader(mvtec_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        train_loader = SyncedDataLoader(rf_train_loader, mvtec_train_loader)
        test_loader = SyncedDataLoader(rf_test_loader, mvtec_test_loader)

        # 3. Feature Extraction & Temporal Analysis
        print("\n3. Model Architecture Initialization...")
        print("   - Feature Extraction: ResNet50")
        print("   - Temporal Analysis: LSTM")

        model = MultiModalArchitecture().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        criterion_recon = nn.MSELoss()
        criterion_anomaly = nn.BCELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 학습
        print("\n4. Model Training...")
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion_recon, criterion_anomaly)
            val_loss = eval_epoch(model, test_loader, criterion_recon)
            scheduler.step()

            print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_multimodal_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # 5. Model Optimization (Performance Metrics)
        print("\n5. Model Optimization - Performance Metrics...")
        model.load_state_dict(torch.load('best_multimodal_model.pth'))
        metrics = compute_performance_metrics(model, test_loader)

        if metrics:
            print(f"Performance Metrics:")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - F1-Score: {metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  - PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
            print(f"  - Optimal Threshold: {metrics['threshold']:.4f}")

        # 6. Manufacturing Process Decision
        print("\n6. Manufacturing Process Decision Ready!")
        print("   Model is ready for real-time defect detection in manufacturing workflows.")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
