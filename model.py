import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def get_model():
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 100) # 修改分类头为100类
    return model
