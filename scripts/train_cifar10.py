"""
Train a CNN classifier for the CIFAR-10 dataset (dense baseline vs MoE variants)
- Supports ResNet-18 with a CIFAR stem
- Standard CIFAR transforms
- SGD + momentum + weight decay
- Supports Dense, SoftMoE, SparseMoE, HardMoE
- tracking best train/test and the epoch they occur (Ma, ETT(Ma), Ga, ETT(Ga))
"""

# --- imports ---
import os, random, numpy as np
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch, torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm
import argparse
from src.moe.data.cifar10_data import build_cifar10_train_test, CIFAR10_STATS


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--ff_width", type=int, default=512)
parser.add_argument("--FF_layer", type=str, default="Dense", choices=["Dense", "SoftMoE", "SparseMoE", "HardMoE"])


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
    FF_WIDTH     = args.ff_width # hidden width of the dense head
    FF_LAYER     = args.FF_layer

    # --- reproducibility / perf ---
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # speed; set False for strict determinism

    # --- dataloaders (train & test only) ---
    train_loader, test_loader, meta = build_cifar10_train_test(
        data_dir="./data",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        augment=True,
        drop_last=False,
    )
    print(f"dataloaders metadata: {meta}")

    pin_memory = (DEVICE == "cuda")  # reuse for non_blocking=True later

    # --- model helpers ---
    # We separate the feature backbone from the dense head
    class FeatureBackbone(nn.Module):
        """ResNet-18 adapted for CIFAR-10, outputs 512-dim features."""
        def __init__(self):
            super().__init__()
            model = resnet18(weights=None)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
            model.maxpool = nn.Identity() #remove the max pooling layer
            model.fc = nn.Identity() #remove the final fully connected layer
            model = model.to(DEVICE)
            self.backbone = model
            self.output_dim = 512

        def forward(self, x):
            return self.backbone(x)

    ##--- Dense Head with width WIDTH (default: 512)---##
    class DenseHead(nn.Module):
        """Dense head with width WIDTH (default: 512)."""
        head_name = "Dense"
        def __init__(self, in_dim=512, width=FF_WIDTH, num_classes=10):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.ReLU(inplace=True), # inplace=True--modify input tensor in place for performance optimization 
                nn.Linear(width, num_classes)
            )

        def forward(self, h, return_gate = False): # h=feature backbone output (B, 512)
            logits = self.fc(h) #y is the output logits (B, 10)
            return (logits, None, None) if return_gate else logits

    # ------- MOE head (handles soft / sparse / hard gating) -------#
    backbone = FeatureBackbone().to(DEVICE)
    assert args.FF_layer == "Dense", "Only Dense head is supported for now"
    head = DenseHead(in_dim=backbone.output_dim, width=FF_WIDTH, num_classes=10).to(DEVICE)

    # ------- Model assembly -------#
    class Classifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head
        def forward(self, x, return_gate=False):
            h = self.backbone(x) # (B, 512)
            return self.head(h, return_gate=return_gate)
    model = Classifier(backbone, head).to(DEVICE)

    # --- generic head suffix for checkpoint names ---
    def head_id_suffix(head):
        """
        Return a short ID string for the head. Works for any head type.
        - Dense head: ''
        - MoE head (if it has these attrs): '-E<num_experts>_K<k>_CF<capacity>'
        """
        parts = []
        for attr, short in [("num_experts", "E"), ("k", "K"), ("capacity_factor", "CF")]:
            if hasattr(head, attr):
                parts.append(f"{short}{getattr(head, attr)}")
        return f"-{'_'.join(parts)}" if parts else ""

    # DEFINE CHECKPOINT PATHS (head-agnostic)4
    # Examples:
    #   Dense:   checkpoints/Dense/W512-S42/
    #   SparseMoE (E=8,K=2,CF=1.25): checkpoints/SparseMoE/W512-S42-E8_K2_CF1.25/
    run_tag  = f"W{FF_WIDTH}-S{SEED}{head_id_suffix(head)}" #this is the checkpoint name
    save_dir = os.path.join("checkpoints", FF_LAYER, run_tag) #checkpoint path
    os.makedirs(save_dir, exist_ok=True) #create the checkpoint path if it doesn't exist

    ckpt_model_path   = os.path.join(save_dir, "model.pt") # output:checkpoints/Dense/W512-S42-E8_K2_CF1.25/model.pt
    ckpt_metrics_path = os.path.join(save_dir, "metrics.pt") # output:checkpoints/Dense/W512-S42-E8_K2_CF1.25/metrics.pt

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
    best_train_acc, best_test_acc = 0.0, 0.0
    best_train_epoch, best_test_epoch = None, None
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    ff_layer = FF_LAYER
    ff_width = FF_WIDTH
 

    # --- checkpoint paths ---
    save_dir = os.path.join("checkpoints", FF_LAYER)   # e.g., checkpoints/Dense
    os.makedirs(save_dir, exist_ok=True)
    ckpt_model_path   = os.path.join(save_dir, f"{FF_WIDTH}_cifar10.pt")
    ckpt_metrics_path = os.path.join(save_dir, f"{FF_WIDTH}_metrics.pt")

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

        # test (instead of val)
        model.eval()
        te_loss_sum, te_correct, te_total = 0.0, 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                logits = model(data)
                loss = criterion(logits, targets)
                te_loss_sum += loss.item()
                c, n = accuracy_from_logits(logits, targets)
                te_correct += c; te_total += n

        test_loss = te_loss_sum / len(test_loader)
        test_acc = te_correct / te_total

        # record + print
        train_losses.append(train_loss); train_accs.append(train_acc)
        test_losses.append(test_loss);   test_accs.append(test_acc)

        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}%")
        
        # track best train/test and the epoch they occur
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_epoch = epoch

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch
            torch.save({"model": model.state_dict(), "test_acc": best_test_acc}, ckpt_model_path)
            print(f"Saved checkpoint: {FF_LAYER} {FF_WIDTH} test_acc={best_test_acc*100:.2f}%")



        # step scheduler
        scheduler.step()

        # save metrics every epoch
        torch.save(
            {"train_losses": train_losses, "train_accs": train_accs,
             "test_losses": test_losses, "test_accs": test_accs},
            ckpt_metrics_path,
        )


    # --- final test evaluation (once more) ---
    if os.path.exists(ckpt_model_path):
        state = torch.load(ckpt_model_path, map_location=DEVICE)
        model.load_state_dict(state["model"])
    else:
        print("No checkpoint found; using current model for final test.")

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
    print(f"[FINAL TEST] loss={test_loss:.4f} acc={test_acc*100:.2f}%")
    
    print("\n=== Summary ===")
    print(f"M_A  (max train acc): {best_train_acc*100:.2f}%  at epoch {best_train_epoch}")
    print(f"G_A  (max test  acc): {best_test_acc*100:.2f}%  at epoch {best_test_epoch}")
    print(f"ETT(M_A) = {best_train_epoch},  ETT(G_A) = {best_test_epoch}")

    # print best train/test and the epoch they occur
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
