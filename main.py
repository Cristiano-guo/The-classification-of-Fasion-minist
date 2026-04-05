"""
Fashion-MNIST 分类：使用 PyTorch + 轻量 CNN，目标准确率 > 90%。
数据集由 torchvision 自动下载到 ./data。
训练结束后在 ./figures 下保存可视化图表，便于撰写实验报告。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys, json, time, os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无界面环境也可保存图片
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FASHION_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
FASHION_NAMES_CN = [
    "T恤/上衣",
    "裤子",
    "套衫",
    "连衣裙",
    "外套",
    "凉鞋",
    "衬衫",
    "运动鞋",
    "包",
    "短靴",
]

# #region agent log
_DBG_LOG = r"e:\深度学习\.cursor\debug.log"
def _dbg(hyp, msg, data=None):
    os.makedirs(os.path.dirname(_DBG_LOG), exist_ok=True)
    entry = {"hypothesisId": hyp, "location": "fashion_mnist_cnn.py", "message": msg, "data": data or {}, "timestamp": int(time.time()*1000)}
    with open(_DBG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
# #endregion


def get_loaders(batch_size: int = 128, data_dir: str = "./data"):
    tfm = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    tfm_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    train_ds = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=tfm
    )
    test_ds = datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=tfm_test
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    return train_loader, test_loader


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def _setup_matplotlib_cn():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_fig_dir(fig_dir: str) -> str:
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def visualize_dataset_samples(
    data_dir: str = "./data", fig_dir: str = "./figures", n_per_class: int = 8
):
    _setup_matplotlib_cn()
    ensure_fig_dir(fig_dir)
    ds = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    by_class = [[] for _ in range(10)]
    for img, label in ds:
        if len(by_class[label]) < n_per_class:
            by_class[label].append(img.squeeze().numpy())
        if all(len(c) >= n_per_class for c in by_class):
            break
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.2, 14))
    fig.suptitle("Fashion-MNIST 各类样本示例", fontsize=14, fontweight="bold")
    for r in range(10):
        for c in range(n_per_class):
            ax = axes[r, c] if n_per_class > 1 else axes[r]
            ax.imshow(by_class[r][c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(
                    f"{FASHION_NAMES_CN[r]}\n({FASHION_NAMES[r]})",
                    fontsize=7,
                    rotation=0,
                    labelpad=40,
                    va="center",
                )
    plt.tight_layout(rect=[0.06, 0, 1, 0.97])
    path = os.path.join(fig_dir, "01_dataset_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {path}")


def plot_training_curves(
    train_losses: list,
    test_accs: list,
    fig_dir: str = "./figures",
    filename: str = "02_training_curves.png",
):
    _setup_matplotlib_cn()
    ensure_fig_dir(fig_dir)
    n = len(train_losses)
    if len(test_accs) != n:
        raise ValueError("train_losses 与 test_accs 长度须一致（均为一轮一个点）")
    # 横轴为训练轮次：第 1 轮 … 第 n 轮
    epoch_ids = np.arange(1, n + 1, dtype=int)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.set_xlabel("训练轮次")
    ax1.set_ylabel("训练损失 (CrossEntropy)", color="tab:blue")
    ax1.plot(epoch_ids, train_losses, "o-", color="tab:blue", label="训练损失", markersize=4)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, n + 0.5)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=min(n, 15), integer=True))

    ax2 = ax1.twinx()
    ax2.set_ylabel("测试集准确率", color="tab:orange")
    ax2.plot(
        epoch_ids,
        [a * 100 for a in test_accs],
        "s-",
        color="tab:orange",
        label="测试准确率 (%)",
        markersize=4,
    )
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0.5, n + 0.5)

    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {path}")


@torch.no_grad()
def plot_confusion_matrix(
    model, loader, device, fig_dir: str = "./figures", filename: str = "03_confusion_matrix.png"
):
    _setup_matplotlib_cn()
    ensure_fig_dir(fig_dir)
    model.eval()
    n = 10
    cm = torch.zeros(n, n, dtype=torch.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1
    cm = cm.numpy().astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    cm_norm = cm / row_sum

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("混淆矩阵（行归一化：每行对应真实类别）", fontsize=12, fontweight="bold")
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([f"{i}\n{FASHION_NAMES_CN[i]}" for i in range(n)], fontsize=7)
    ax.set_yticklabels([f"{i}\n{FASHION_NAMES_CN[i]}" for i in range(n)], fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("真实类别")
    ax.set_xlabel("预测类别")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="比例")

    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=6,
            )
    plt.tight_layout()
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {path}")


def _denormalize_fashion(tensor_chw: torch.Tensor) -> np.ndarray:
    """反归一化到 [0,1] 便于 imshow。"""
    mean, std = 0.2860, 0.3530
    x = tensor_chw.detach().cpu().squeeze().numpy() * std + mean
    return np.clip(x, 0.0, 1.0)


@torch.no_grad()
def visualize_predictions(
    model,
    loader,
    device,
    fig_dir: str = "./figures",
    filename: str = "04_prediction_samples.png",
    n_correct: int = 8,
    n_wrong: int = 8,
):
    _setup_matplotlib_cn()
    ensure_fig_dir(fig_dir)
    model.eval()
    correct_imgs, correct_true, correct_pred = [], [], []
    wrong_imgs, wrong_true, wrong_pred = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        for i in range(x.size(0)):
            if pred[i] == y[i]:
                if len(correct_imgs) < n_correct:
                    correct_imgs.append(x[i])
                    correct_true.append(y[i].item())
                    correct_pred.append(pred[i].item())
            else:
                if len(wrong_imgs) < n_wrong:
                    wrong_imgs.append(x[i])
                    wrong_true.append(y[i].item())
                    wrong_pred.append(pred[i].item())
        if len(correct_imgs) >= n_correct and len(wrong_imgs) >= n_wrong:
            break

    rows = 2
    cols = max(n_correct, n_wrong)
    fig, axes = plt.subplots(rows, cols, figsize=(max(cols, 1) * 1.4, 3.8))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)
    titles = ["预测正确样例", "预测错误样例（真实/预测见子图标题）"]
    for r, (imgs, trues, preds, title) in enumerate(
        zip(
            [correct_imgs, wrong_imgs],
            [correct_true, wrong_true],
            [correct_pred, wrong_pred],
            titles,
        )
    ):
        for c in range(cols):
            ax = axes[r, c]
            if c < len(imgs):
                ax.imshow(_denormalize_fashion(imgs[c]), cmap="gray", vmin=0, vmax=1)
                t, p = trues[c], preds[c]
                if r == 0:
                    ax.set_title(
                        f"真:{FASHION_NAMES_CN[t][:4]}\n预:{FASHION_NAMES_CN[p][:4]}",
                        fontsize=7,
                    )
                else:
                    ax.set_title(
                        f"真:{FASHION_NAMES_CN[t]}\n预:{FASHION_NAMES_CN[p]}",
                        fontsize=7,
                        color="red" if t != p else "black",
                    )
            ax.axis("off")
        axes[r, 0].set_ylabel(title, fontsize=9, fontweight="bold")
    fig.suptitle("测试集预测样例", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {path}")


def train(
    epochs: int = 25,
    lr: float = 0.001,
    batch_size: int = 128,
    data_dir: str = "./data",
    target_acc: float = 0.90,
    visualize: bool = True,
    fig_dir: str = "./figures",
):
    # #region agent log
    _dbg("env", "python_executable", {"path": sys.executable})
    _dbg("env", "torch_details", {
        "version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "torch_file": torch.__file__,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    })
    # #endregion

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if getattr(torch.version, "cuda", None) is None:
            print(
                "提示: 当前 PyTorch 为 CPU 构建（不含 CUDA）。"
                "请到 https://pytorch.org 选择带 CUDA 的安装命令重装 torch/torchvision，"
                "并确保已安装与版本匹配的 NVIDIA 驱动。"
            )

    # #region agent log
    _dbg("device", "device_selected", {"device": str(device)})
    # #endregion

    print(f"设备: {device}")

    if visualize:
        visualize_dataset_samples(data_dir=data_dir, fig_dir=fig_dir)

    train_loader, test_loader = get_loaders(batch_size, data_dir)
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses: list = []
    test_accs: list = []
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss = running_loss / max(n_batches, 1)
        test_acc = evaluate(model, test_loader, device)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        best_acc = max(best_acc, test_acc)
        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"test_acc={test_acc * 100:.2f}%  "
            f"best={best_acc * 100:.2f}%"
        )
        if test_acc >= target_acc:
            print(f"已达到目标准确率 >= {target_acc * 100:.0f}%")
            break

    print(f"最终测试集最佳准确率: {best_acc * 100:.2f}%")
    if best_acc < target_acc:
        print("提示: 可适当增加 epochs 或 batch_size，或检查是否使用 GPU。")

    if visualize and train_losses:
        plot_training_curves(train_losses, test_accs, fig_dir=fig_dir)
        plot_confusion_matrix(model, test_loader, device, fig_dir=fig_dir)
        visualize_predictions(model, test_loader, device, fig_dir=fig_dir)

    return best_acc


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    train()
