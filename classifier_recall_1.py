import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os

def load_data(data_dir, batch_size=32, train_size=0.7, valid_size=0.2):
    """
    데이터를 train/valid/test로 분할하여 로드
    """
    # 데이터 변환 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 로드
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # 데이터 분할 (train:valid:test = 7:2:1)
    num_total = len(dataset)
    indices = list(range(num_total))
    np.random.seed(42)  # 재현 가능한 결과를 위해
    np.random.shuffle(indices)
    
    train_split = int(train_size * num_total)
    valid_split = int((train_size + valid_size) * num_total)
    
    train_idx = indices[:train_split]
    valid_idx = indices[train_split:valid_split]
    test_idx = indices[valid_split:]
    
    # 서브셋 생성
    train_subset = Subset(dataset, train_idx)
    valid_subset = Subset(dataset, valid_idx)
    test_subset = Subset(dataset, test_idx)
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, dataset.classes

def prepare_data_for_xgboost(loader, max_samples=None):
    """
    PyTorch DataLoader에서 XGBoost용 데이터로 변환
    메모리 효율성을 위해 샘플 수 제한 옵션 추가
    """
    X = []
    y = []
    sample_count = 0
    
    for images, labels in loader:
        # 이미지를 평탄화 (batch_size, channels*height*width)
        images_flat = images.view(images.size(0), -1).numpy()
        labels_np = labels.numpy()
        
        X.extend(images_flat)
        y.extend(labels_np)
        
        sample_count += len(images_flat)
        if max_samples and sample_count >= max_samples:
            break
    
    return np.array(X), np.array(y)

def train_xgboost_model(X_train, y_train, X_valid, y_valid, num_classes):
    """
    XGBoost 모델 학습
    """
    # 레이블 정보 확인 및 수정
    print(f"Original label range - Train: {y_train.min()}-{y_train.max()}, Valid: {y_valid.min()}-{y_valid.max()}")
    print(f"Unique labels - Train: {np.unique(y_train)}, Valid: {np.unique(y_valid)}")
    
    # 실제 클래스 개수 재계산 (train과 valid 합쳐서)
    all_labels = np.concatenate([y_train, y_valid])
    unique_labels = np.unique(all_labels)
    actual_num_classes = len(unique_labels)
    
    print(f"Dataset classes: {num_classes}, Actual unique classes: {actual_num_classes}")
    print(f"Unique label values: {unique_labels}")
    
    # 레이블이 0부터 연속적이지 않은 경우 재인코딩
    if not (unique_labels == np.arange(len(unique_labels))).all():
        print("Re-encoding labels to be continuous from 0...")
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        print(f"Label mapping: {label_mapping}")
        
        # 레이블 재인코딩
        y_train_encoded = np.array([label_mapping[label] for label in y_train])
        y_valid_encoded = np.array([label_mapping[label] for label in y_valid])
        
        print(f"Encoded label range - Train: {y_train_encoded.min()}-{y_train_encoded.max()}")
        print(f"Encoded unique labels: {np.unique(y_train_encoded)}")
    else:
        y_train_encoded = y_train
        y_valid_encoded = y_valid
    
    # XGBoost 모델 파라미터 설정
    if actual_num_classes == 2:
        params = {
            'objective': 'binary:logistic',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
    else:
        params = {
            'objective': 'multi:softprob',
            'num_class': actual_num_classes,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1
        }
    
    print(f"Training XGBoost with {len(X_train)} samples, {actual_num_classes} classes")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"XGBoost parameters: {params}")
    
    # DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
    dvalid = xgb.DMatrix(X_valid, label=y_valid_encoded)
    
    # 학습 파라미터 설정
    num_rounds = 100
    evallist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    # 모델 학습
    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evallist,
        early_stopping_rounds=10,
        verbose_eval=10  # 10번마다 출력
    )
    
    return model

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    모델 평가
    """
    # 예측
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    
    # 다중분류의 경우 확률에서 클래스 추출
    if len(y_pred_proba.shape) > 1:
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # 이진분류의 경우
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # 분류 보고서 출력
    print('\nClassification Report:')
    target_names = class_names if class_names else None
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return accuracy, y_pred

def main():
    # 설정
    data_dir = './data/data'  # 데이터 디렉토리 경로
    batch_size = 64
    max_samples_per_split = 5000  # 메모리 절약을 위한 샘플 수 제한
    
    print("Loading data...")
    try:
        train_loader, valid_loader, test_loader, class_names = load_data(
            data_dir, batch_size=batch_size
        )
        print(f"Classes: {class_names}")
        print(f"Number of classes: {len(class_names)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check if the data directory exists and contains image folders.")
        return
    
    # XGBoost용 데이터 준비
    print("Preparing training data...")
    X_train, y_train = prepare_data_for_xgboost(train_loader, max_samples_per_split)
    
    print("Preparing validation data...")
    X_valid, y_valid = prepare_data_for_xgboost(valid_loader, max_samples_per_split)
    
    print("Preparing test data...")
    X_test, y_test = prepare_data_for_xgboost(test_loader, max_samples_per_split)
    
    print(f"Data shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    
    # 레이블 정보 출력
    print(f"\nLabel information:")
    print(f"Train labels: min={y_train.min()}, max={y_train.max()}, unique={np.unique(y_train)}")
    print(f"Valid labels: min={y_valid.min()}, max={y_valid.max()}, unique={np.unique(y_valid)}")
    print(f"Test labels: min={y_test.min()}, max={y_test.max()}, unique={np.unique(y_test)}")
    
    # 모델 학습
    print("\nTraining XGBoost model...")
    model = train_xgboost_model(X_train, y_train, X_valid, y_valid, len(class_names))
    
    # 모델 평가 (테스트 데이터도 같은 방식으로 레이블 인코딩 필요)
    print("\nEvaluating model on test set...")
    
    # 테스트 데이터의 레이블도 동일하게 인코딩해야 함
    all_labels = np.concatenate([y_train, y_valid, y_test])
    unique_labels = np.unique(all_labels)
    
    if not (unique_labels == np.arange(len(unique_labels))).all():
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        y_test_encoded = np.array([label_mapping[label] for label in y_test])
    else:
        y_test_encoded = y_test
    
    accuracy, predictions = evaluate_model(model, X_test, y_test_encoded, class_names)
    
    # 모델 저장
    model_path = 'xgboost_image_classifier.json'
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Feature importance 출력
    print("\nTop 10 Feature Importances:")
    importance = model.get_score(importance_type='weight')
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_importance[:10]):
            print(f"{i+1}. {feature}: {score}")
    else:
        print("No feature importance available.")

if __name__ == '__main__':
    main()