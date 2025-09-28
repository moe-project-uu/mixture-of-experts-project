"""
Train a CNN classifier for the CIFAR-10 dataset (dense baseline vs MoE)
- Supports ResNet-18 or ResNet-50 with a CIFAR stem
- Proper train/val split (45k/5k) and standard CIFAR transforms
- SGD + momentum + weight decay
"""

# --- imports ---
import os, random, numpy as np
import torch, torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--resnet_depth", type=int, default=18, choices=[18, 50])

def main(args):
    # --- hyperparameters ---
    BATCH_SIZE   = args.batch_size
    NUM_WORKERS  = args.num_workers
    EPOCHS       = args.epochs
    LR           = args.learning_rate
    MOMENTUM     = args.momentum
    WEIGHT_DECAY = args.weight_decay
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    SEED         = 42
    ARCH_DEPTH   = args.resnet_depth  # 18 or 50

    # --- reproducibility / perf ---
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # speed; set False for strict determinism

    # --- CIFAR transforms (AUGS BEFORE ToTensor/Normalize) ---
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    val_test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # --- datasets ---
    root = "./data"
    train_full = torchvision.datasets.CIFAR10(root=root, train=True,  download=True, transform=train_tf)
    test_set   = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=val_test_tf)

    # split train into train/val (45k/5k)
    val_size = 5000
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    # --- dataloaders ---
    pin_memory = (DEVICE == "cuda")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_memory)

    # --- model helpers ---
    def resnet18_cifar(num_classes=10):
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR stem
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(512, num_classes)
        return m

    def resnet50_cifar(num_classes=10):
        m = resnet50(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(2048, num_classes)
        return m

    # pick model
    if ARCH_DEPTH == 18:
        model = resnet18_cifar(num_classes=10).to(DEVICE)
        arch_name = "resnet18"
    else:
        model = resnet50_cifar(num_classes=10).to(DEVICE)
        arch_name = "resnet50"

    # --- loss & optimizer & (optional) scheduler ---
    criterion = nn.CrossEntropyLoss()  # you can try label_smoothing=0.1
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)  # good for 50 epochs

    # --- metrics helpers ---
    def accuracy_from_logits(logits, targets):
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct, targets.size(0)

    # --- training loop ---
    best_val_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_model_path = f"checkpoints/{arch_name}_cifar10.pt"
    ckpt_metrics_path = f"checkpoints/{arch_name}_metrics.pt"

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for data, targets in pbar:
            data, targets = data.to(DEVICE, non_blocking=pin_memory), targets.to(DEVICE, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item()
            c, n = accuracy_from_logits(logits, targets)
            tr_correct += c; tr_total += n

        train_loss = tr_loss_sum / len(train_loader)
        train_acc = tr_correct / tr_total

        # validate
        model.eval()
        va_loss_sum, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                logits = model(data)
                loss = criterion(logits, targets)
                va_loss_sum += loss.item()
                c, n = accuracy_from_logits(logits, targets)
                va_correct += c; va_total += n

        val_loss = va_loss_sum / len(val_loader)
        val_acc = va_correct / va_total

        # record + print
        train_losses.append(train_loss); train_accs.append(train_acc)
        val_losses.append(val_loss);     val_accs.append(val_acc)

        print(f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        # step scheduler
        scheduler.step()

        # save metrics every epoch
        torch.save(
            {"train_losses": train_losses, "train_accs": train_accs,
            "val_losses": val_losses, "val_accs": val_accs},
            ckpt_metrics_path,
        )

        # save best model on val
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "val_acc": best_val_acc}, ckpt_model_path)
            print(f"Saved checkpoint: {arch_name} val_acc={best_val_acc*100:.2f}%")

    # --- final test evaluation (once) ---
    model.load_state_dict(torch.load(ckpt_model_path, map_location=DEVICE)["model"])
    model.eval()
    te_correct, te_total, te_loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            logits = model(data)
            loss = criterion(logits, targets)
            te_loss_sum += loss.item()
            c, n = accuracy_from_logits(logits, targets)
            te_correct += c; te_total += n

    test_loss = te_loss_sum / len(test_loader)
    test_acc  = te_correct / te_total
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc*100:.2f}%")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
