import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------
# 1. ViT 模型定义（适配FashionMNIST）
# -------------------
class ViTForFashionMNIST(nn.Module):
    def __init__(self, image_size=28, patch_size=4, num_classes=10,
                 dim=64, depth=4, heads=4, mlp_dim=128):
        super().__init__()

        # 计算patch数量
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size ** 2  # 输入通道为1（灰度图）

        # Patch嵌入层
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                activation="gelu",
                batch_first=True
            ),
            num_layers=depth
        )

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # 输入形状: (B, 1, 28, 28)
        x = self.patch_embed(x)  # (B, dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        x += self.pos_embed

        # Transformer处理
        x = self.transformer(x)

        # 取CLS token输出
        cls_output = x[:, 0]

        # 分类
        return self.mlp_head(cls_output)


# -------------------
# 2. 数据预处理
# -------------------
def get_dataloaders(batch_size=64):
    # 数据增强（针对小数据集）
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToTensor(),
        v2.Normalize((0.5,), (0.5,))
    ])

    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.5,), (0.5,))
    ])

    # 加载数据集
    train_set = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_set = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# -------------------
# 3. 训练配置
# -------------------
def train_model(model, train_loader, test_loader, num_epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = correct / total
        history['test_acc'].append(epoch_acc)

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_vit_fashionmnist.pth')

        # 更新学习率
        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Train Loss: {epoch_loss:.4f} | '
              f'Test Acc: {epoch_acc:.4f}')

    # 训练曲线可视化
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], label='Test Acc', color='orange')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

    return model


# -------------------
# 4. 主函数
# -------------------
if __name__ == "__main__":
    # 初始化模型（适配FashionMNIST的轻量ViT）
    model = ViTForFashionMNIST(
        image_size=28,
        patch_size=4,
        dim=64,  # 适当减小维度
        depth=4,  # 减少Transformer层数
        heads=4,
        mlp_dim=128
    ).to(device)

    print(f"模型参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 获取数据加载器
    train_loader, test_loader = get_dataloaders(batch_size=128)  # 根据显存调整

    # 开始训练
    trained_model = train_model(model, train_loader, test_loader, num_epochs=15)