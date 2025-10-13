"""
Train a CNN classifier for the CIFAR-10 dataset (dense baseline vs MoE variants)
- Supports ResNet-18 with a CIFAR stem
- Standard CIFAR transforms
- SGD + momentum + weight decay
- Supports Dense, SoftMoE, SparseMoE, HardMoE
- tracking best train/val/test and the epoch they occur (Ma, ETT(Ma), Ga, ETT(Ga))
"""

# --- imports ---
import os, random, numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from moe.data.cifar10_data import build_cifar10_train_val_test, CIFAR10_STATS
#project imports
from moe.models.backbones import FeatureBackbone
from moe.heads.factory import build_head

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--ff_width", type=int, default=512)
parser.add_argument("--num_experts", type=int, default=4)
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--dropout_p", type=float, default=0.1)
parser.add_argument("--hidden_mult", type=float, default=2)
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
    NUM_EXPERTS  = args.num_experts
    TEMPERATURE  = args.temperature
    DROPOUT_P    = args.dropout_p
    HIDDEN_MULT  = args.hidden_mult

    # --- checkpoint path (general) ---
    if FF_LAYER == "Dense":
        run_tag = f"E{EPOCHS}"
    else:
        run_tag = f"E{EPOCHS}-X{NUM_EXPERTS}"  # X = num_experts

    ckpt_dir = os.path.join("checkpoints", FF_LAYER, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_model_path = os.path.join(ckpt_dir, "model.pt")

    # --- reproducibility / performance ---
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # speed; set False for strict determinism

    # --- dataloaders (train, val, test) ---
    train_loader, val_loader, test_loader, meta = build_cifar10_train_val_test(
        data_dir="./data",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        augment=True,
        drop_last=False,
        val_ratio=0.1,
        seed=SEED,
    )
    print(f"dataloaders metadata: {meta}")

    pin_memory = (DEVICE == "cuda")  # reuse for non_blocking=True later

    # --- build model (near where you had backbone/head/classifier) ---
    backbone = FeatureBackbone().to(DEVICE)
    if FF_LAYER == "Dense":
        head = build_head(
            FF_LAYER,                    # "Dense" for now; Soft/Sparse/Hard later
            in_dim=backbone.output_dim,       #512 
            width=FF_WIDTH,                   # only used by Dense
            num_classes=10,
        ).to(DEVICE)
    elif FF_LAYER == "SoftMoE":
        head = build_head(
            FF_LAYER,                    # "SoftMoE" for now; Sparse/Hard later
            in_dim=backbone.output_dim,       #512 
            num_classes=10,
            num_experts=NUM_EXPERTS,
            hidden_mult=HIDDEN_MULT,
            temperature=TEMPERATURE,
            dropout_p=DROPOUT_P,
        ).to(DEVICE)
    else:
        raise NotImplementedError(f"{FF_LAYER} not implemented yet")

    class Classifier(nn.Module):
        def __init__(self, backbone, head): 
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x, return_gate=False):
            h = self.backbone(x)                      # (B, 512)
            return self.head(h, return_gate=return_gate)

    model = Classifier(backbone, head).to(DEVICE)


    # --- loss & optimizer & scheduler ---
    criterion = nn.CrossEntropyLoss()  # you can try label_smoothing=0.1
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)  # good for 50 epochs

    # --- metrics helpers ---
    def accuracy_from_logits(logits, targets):
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct, targets.size(0)

    # --- training loop ---
    best_train_acc, best_val_acc = 0.0, 0.0   # Ma, Ga (Ga = best val acc)
    best_train_epoch, best_val_epoch = None, None   # ETT(Ma), ETT(Ga)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []   # for plotting

    # --- gating metrics history (SoftMoE only) ---
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        # SoftMoE-specific:
        "util_per_epoch": [],   # utilization per epoch.. list of np arrays shape (E,)
        "entropy_per_epoch": [] # entropy per epoch.. list of floats
    }

    for epoch in range(1, EPOCHS + 1):
        # reset per-epoch accumulators for gating stats
        if FF_LAYER == "SoftMoE":
            util_sum = torch.zeros(NUM_EXPERTS, device=DEVICE) #utilization sum
            ent_sum = 0.0 #entropy sum
            count_samples = 0 
        # train
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for data, targets in pbar:
            data, targets = data.to(DEVICE, non_blocking=pin_memory), targets.to(DEVICE, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            ##################
            if FF_LAYER == "Dense":
                logits = model(data, return_gate=False)
            elif FF_LAYER == "SoftMoE":
                logits, probs, _, _ = model(data, return_gate=True) #sel_idx, aux_loss both set to none for now
                # probs: (B, E)
                B = probs.size(0)
                util_sum += probs.sum(dim=0)  # sum over batch for each expert
                # per-sample entropy: -(p * log p).sum(-1), then sum over batch
                ent_batch = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1)  # (B,)
                ent_sum += ent_batch.sum().item()
                count_samples += B

            else:
                raise NotImplementedError
                #logits, probs, sel_idx, aux_loss = model(data, return_gate=True) 
            ##################
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item()
            c, n = accuracy_from_logits(logits, targets)
            tr_correct += c; tr_total += n

        train_loss = tr_loss_sum / len(train_loader)
        train_acc = tr_correct / tr_total
        # record generic learning curves
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # record SoftMoE gating stats per epoch
        if FF_LAYER == "SoftMoE" and count_samples > 0:
            util_epoch = (util_sum / count_samples).detach().cpu().numpy()  # shape (E,)
            H_epoch = ent_sum / count_samples 
            history["util_per_epoch"].append(util_epoch)
            history["entropy_per_epoch"].append(H_epoch)


        # --- validation ---
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                ##################
                if FF_LAYER == "Dense":
                    logits = model(data, return_gate=False)
                elif FF_LAYER == "SoftMoE":
                    logits, probs, _, _ = model(data, return_gate=True) #sel_idx, aux_loss both set to none for now
                else:
                    raise NotImplementedError
                    #logits, probs, sel_idx, aux_loss = model(data, return_gate=True) 
                ##################
                
                loss = criterion(logits, targets)
                val_loss_sum += loss.item()
                c, n = accuracy_from_logits(logits, targets)
                val_correct += c; val_total += n

        val_loss = val_loss_sum / len(val_loader)
        val_acc = val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # record per epoch stats
        train_losses.append(train_loss); train_accs.append(train_acc)
        val_losses.append(val_loss);     val_accs.append(val_acc)

        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        
        # track best train/val and the epoch they occur (Ma, Ga, ETT(Ma), ETT(Ga))
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_epoch = epoch

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            torch.save({"model": model.state_dict(), "val_acc": best_val_acc}, ckpt_model_path)
            print(f"Saved checkpoint: {FF_LAYER} val_acc={best_val_acc*100:.2f}%")


        # step scheduler
        scheduler.step()

    # save metrics for plotting later (optional)
    torch.save(history, os.path.join(ckpt_dir, "metrics.pt"))

    # --- final test evaluation (once more, on best-val model) ---
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
    if FF_LAYER == "Dense":
        print(f"Width {FF_WIDTH}: ")
        print(f"M_A  (max train acc): {best_train_acc*100:.2f}%  at epoch {best_train_epoch}")
        print(f"G_A  (max val  acc): {best_val_acc*100:.2f}%  at epoch {best_val_epoch}")
        print(f"ETT(M_A) = {best_train_epoch},  ETT(G_A) = {best_val_epoch}")
    elif FF_LAYER == "SoftMoE":
        expert_hidden = int(HIDDEN_MULT * backbone.output_dim)
        total_width   = NUM_EXPERTS * expert_hidden
        print(f"num_experts {NUM_EXPERTS}")
        print(f"expert_width {expert_hidden}")
        print(f"total_width  {total_width}")
        print(f"M_A  (max train acc): {best_train_acc*100:.2f}%  at epoch {best_train_epoch}")
        print(f"G_A  (max val  acc): {best_val_acc*100:.2f}%  at epoch {best_val_epoch}")
        print(f"ETT(M_A) = {best_train_epoch},  ETT(G_A) = {best_val_epoch}")
    else:
        raise NotImplementedError
    
    return history
#### -----  End of main function ----- ####


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
