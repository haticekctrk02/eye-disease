import timm
import torch.nn as nn

def create_model(num_classes, pretrained=True, dropout=0.4):
    """EfficientNet-B0 tabanlı son katmanları özelleştiren model oluşturma."""
    model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
    # timm üzerinde zaten classifier konfigüre, ek dropout istenirse farklı şekilde değiştirilebilir
    if dropout and hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    return model
