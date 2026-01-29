import os
import argparse
from pathlib import Path
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from model import create_model


def get_loaders(data_dir, img_size=224, batch_size=32, num_workers=4):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dir = Path(data_dir)
    # Assumes data_dir contains class subfolders and we'll split manually
    full_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    classes = full_dataset.classes
    n = len(full_dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = torch.utils.data.random_split(full_dataset, [n_train, n_val, n_test])
    # Replace transforms for val/test
    val_set.dataset.transform = val_transforms
    test_set.dataset.transform = val_transforms

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, classes


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, classes = get_loaders(args.data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers)
    num_classes = len(classes)
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    best_f1 = 0.0
    patience = 6
    counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader.dataset)
        train_f1 = f1_score(all_labels, all_preds, average='macro')

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                val_preds.extend(preds.tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())
        val_loss = val_loss / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_f1)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_f1={train_f1:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}, val_acc={val_acc:.4f}")

        # Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'classes': classes}, 'models/best.pth')
            # Ayrıca sınıf listesini düz metin olarak kaydet
            with open('models/classes.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(classes))
            print('Saved best model and classes.txt')
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping')
                break

    # Final test evaluation
    print('Running final evaluation on test set...')
    from eval import evaluate
    evaluate('models/best.pth', test_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--data-dir', type=str, default='../Original Dataset', help='Path to dataset root (class subfolders)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    train(args)
