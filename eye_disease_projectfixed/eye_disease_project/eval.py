import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


def evaluate(model_path, dataloader, device):
    ckpt = torch.load(model_path, map_location=device)
    model_state = ckpt['model_state_dict']
    classes = ckpt.get('classes', None)

    # Lazy import model creation to avoid circular import
    from model import create_model
    num_classes = len(classes) if classes else 10
    model = create_model(num_classes, pretrained=False)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            p = outputs.argmax(dim=1).detach().cpu().numpy()
            preds.extend(p.tolist())
            labels.extend(targets.numpy().tolist())

    report = classification_report(labels, preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(labels, preds)

    os.makedirs('reports', exist_ok=True)
    # Save classification report
    import json
    with open('reports/classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

    print('Classification report ve confusion matrix kaydedildi: reports/')
    # Basic print
    print('Macro F1 (ortalama):', report['macro avg']['f1-score'])
    return report, cm
