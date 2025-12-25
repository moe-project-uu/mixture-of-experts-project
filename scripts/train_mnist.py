"""
Train a CNN classifier for the MNIST dataset (dense baseline vs MoE variants)
"""

# --- imports ---
import os, random, json, numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from moe.data.mnist_data import build_mnist_train_val_test, MNIST_STATS
# project imports
from moe.models.backbones import MNISTFeatureBackbone
from moe.heads.factory import build_head
from moe.utils.losses import softmoe_load_balance



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
parser.add_argument("--gate_input_dropout", type=float, default=0.1)
parser.add_argument("--gate_logits_dropout", type=float, default=0.1)
parser.add_argument("--hidden_mult", type=float, default=2)
parser.add_argument("--FF_layer", type=str, default="Dense", choices=["Dense", "SoftMoE", "SparseMoE", "HardMoE"])
parser.add_argument("--softmoe_load_balance", type=bool, default=False)
parser.add_argument("--softmoe_load_balance_coef", type=float, default=0.05)
parser.add_argument("--sparsemoe_importance_coef", type=float, default=0.1)
parser.add_argument("--sparsemoe_load_coef", type=float, default=0.1)
parser.add_argument("--sparsemoe_k", type=int, default=2)
parser.add_argument(
    "--ckpt_root",
    type=str,
    default="checkpoints",
    help="Root directory to store checkpoints. "
         "In Colab, set this to a Google Drive path like "
         "'/content/drive/MyDrive/moe_project/checkpoints'.",
)



def main(args):
    # --- hyperparameters ---
    DATASET      = "mnist"
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
    GATE_INPUT_DROPOUT = args.gate_input_dropout
    GATE_LOGITS_DROPOUT = args.gate_logits_dropout
    SPARSEMOE_IMPORTANCE_COEF = args.sparsemoe_importance_coef
    SPARSEMOE_LOAD_COEF = args.sparsemoe_load_coef
    SPARSEMOE_K = args.sparsemoe_k
    CKPT_ROOT = args.ckpt_root

    # --- checkpoint path (general) ---
    if FF_LAYER == "Dense":
        run_tag = f"E{EPOCHS}"
    elif FF_LAYER == "SparseMoE":
        run_tag = f"E{EPOCHS}-X{NUM_EXPERTS}-K{SPARSEMOE_K}"
    else:
        run_tag = f"E{EPOCHS}-X{NUM_EXPERTS}"

    ckpt_dir = os.path.join(CKPT_ROOT, DATASET, FF_LAYER, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_model_path = os.path.join(ckpt_dir, "model.pt")

    # --- save json run summary for later reconstruction (Hessian script, etc.) ---
    summary = {
        "FF_layer": FF_LAYER,
        "ff_width": FF_WIDTH,
        "num_experts": NUM_EXPERTS,
        "hidden_mult": HIDDEN_MULT,
        "temperature": TEMPERATURE,
        "dropout_p": DROPOUT_P,
        "gate_input_dropout": GATE_INPUT_DROPOUT,
        "gate_logits_dropout": GATE_LOGITS_DROPOUT,
        "sparsemoe_k": SPARSEMOE_K,
        "sparsemoe_importance_coef": SPARSEMOE_IMPORTANCE_COEF,
        "sparsemoe_load_coef": SPARSEMOE_LOAD_COEF,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "val_ratio": 0.1,      
        "ckpt_dir": ckpt_dir,
        "ckpt_model_path": ckpt_model_path,
    }
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    #---end of save run summary---

    # --- reproducibility / performance ---
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # speed; set False for strict determinism

    # --- dataloaders (train, val, test) ---
    train_loader, val_loader, test_loader, meta = build_mnist_train_val_test(
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

    # --- build model  ---
    backbone = MNISTFeatureBackbone().to(DEVICE)
    if FF_LAYER == "Dense":
        head = build_head(
            FF_LAYER,                    # "Dense"
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
            gate_input_dropout=GATE_INPUT_DROPOUT,
            gate_logits_dropout=GATE_LOGITS_DROPOUT,
        ).to(DEVICE)
    elif FF_LAYER == "SparseMoE":
        head = build_head(
            FF_LAYER,                    # "SparseMoE"
            in_dim=backbone.output_dim,       #512 
            num_classes=10,
            num_experts=NUM_EXPERTS,
            hidden_mult=HIDDEN_MULT,
            temperature=TEMPERATURE,
            dropout_p=DROPOUT_P,
            gate_input_dropout=GATE_INPUT_DROPOUT,
            gate_logits_dropout=GATE_LOGITS_DROPOUT,
            importance_coef=SPARSEMOE_IMPORTANCE_COEF,
            load_coef=SPARSEMOE_LOAD_COEF,
            k=SPARSEMOE_K, #default k=2 for noisy top-k sparse moe (set to 1 for hardmoe)
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
        """output: correct, total (so that we can calculate 
        the accuracy = correct/total)
        """
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct, targets.size(0)

    # --- training loop ---
    best_train_acc, best_val_acc = 0.0, 0.0   # Ma, Ga (Ga = best val acc)
    best_train_epoch, best_val_epoch = None, None   # ETT(Ma), ETT(Ga)

    # --- gating metrics history (SoftMoE only) ---
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "util_per_epoch": [],   # utilization per epoch.. list of np arrays shape (E,)
        "entropy_per_epoch": [] # entropy per epoch.. list of floats
    }
    if FF_LAYER == "SparseMoE":
        history["load_per_epoch"] = [] # list of np arrays shape (E,) to track count of images per expert in the batch per epoch

    for epoch in range(1, EPOCHS + 1):
        # reset per-epoch accumulators for gating stats
        if FF_LAYER in ["SoftMoE", "SparseMoE"]:
            util_sum = torch.zeros(NUM_EXPERTS, device=DEVICE) #utilization sum (probabilities sum)
            ent_sum = 0.0 #entropy sum
            count_samples = 0 
            if FF_LAYER == "SparseMoE":
                load_sum = torch.zeros(NUM_EXPERTS, device=DEVICE)
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
                logits, probs, _, aux_loss = model(data, return_gate=True) #sel_idx, aux_loss both set to none for now
                # probs: (B, E)
                B = probs.size(0)
                util_sum += probs.sum(dim=0)  # sum over batch for each expert
                # per-sample entropy: -(p * log p).sum(-1), then sum over batch
                ent_batch = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1)  # (B,)
                ent_sum += ent_batch.sum().item()
                count_samples += B
            elif FF_LAYER == "SparseMoE":
                logits, probs, sel_idx, aux_loss = model(data, return_gate=True) #sel_idx, aux_loss both set to none for now
                # probs: (B, E)
                B = probs.size(0)
                util_sum += probs.sum(dim=0)  # sum over batch for each expert
                # per-sample entropy: -(p * log p).sum(-1), then sum over batch
                ent_batch = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1)  # (B,)
                ent_sum += ent_batch.sum().item()
                load_sum += (probs > 0).float().sum(dim=0) # number of images per expert in the batch
                count_samples += B

            else:
                raise NotImplementedError
                #logits, probs, sel_idx, aux_loss = model(data, return_gate=True) 
            
            ### -- ADD RELEVANT AUXILIARY LOSS TERMS HERE FOR LOAD BALANCING -- ###
            if FF_LAYER == "SoftMoE" and args.softmoe_load_balance:
                loss = criterion(logits, targets) + softmoe_load_balance(probs, NUM_EXPERTS, coef=args.softmoe_load_balance_coef)
            elif FF_LAYER == "SparseMoE":
                #aux_loss is the sparse moe auxiliary loss which is the sum of the importance 
                # and load losses (returned by the sparse moe head)
                loss = criterion(logits, targets) + (aux_loss if (aux_loss is not None) else 0.0)
            else:
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
        if FF_LAYER in ["SoftMoE", "SparseMoE"] and count_samples > 0:
            util_epoch = (util_sum / count_samples).detach().cpu().numpy()  # shape (E,)
            H_epoch = ent_sum / count_samples 
            history["util_per_epoch"].append(util_epoch)
            history["entropy_per_epoch"].append(H_epoch)

            #fraction of images that routed to each expert (Sparse)
            # or expected fraction (Soft)

            if FF_LAYER == "SparseMoE":
                load_epoch = (load_sum / count_samples).detach().cpu().numpy()
                history["load_per_epoch"].append(load_epoch)



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
                    logits, probs, _, aux_loss = model(data, return_gate=True) #sel_idx, aux_loss both set to none for now
                elif FF_LAYER == "SparseMoE":
                    logits, probs, sel_idx, aux_loss = model(data, return_gate=True)
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
            #SAVE BEST MODEL 
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

    # --- per-class gating probabilities Test time (SoftMoE and SparseMoE) ---
    # get class_expert_mean: np.ndarray of shape (num_classes, num_experts) for input to plotting functions
    if FF_LAYER in ["SoftMoE", "SparseMoE"]:
        model.eval()
        num_classes = 10
        class_names = ["0","1","2","3","4","5","6","7","8","9"]  # for MNIST

        class_prob_sums = torch.zeros(num_classes, NUM_EXPERTS, device=DEVICE) # (10, E)
        class_counts    = torch.zeros(num_classes, device=DEVICE) # (10,)

        with torch.no_grad():
            #now we're measuring the per-class gating probabilities for the test set
            for data, targets in test_loader: #data is (B, 3, 32, 32), targets is (B,)
                data   = data.to(DEVICE)
                labels = targets.to(DEVICE)  # (B,)
                logits, probs, _, _ = model(data, return_gate=True)  # probs: (B, E)

                # accumulate probability sums per class in a vectorized way
                # one-hot: (B, 10)
                one_hot = torch.zeros(probs.size(0), num_classes, device=DEVICE, dtype=probs.dtype) # (B, 10)
                one_hot.scatter_(1, labels.unsqueeze(1), 1.0) # set correct class to 1


                # (10, B) @ (B, E) -> (10, E): sum probs for samples of each class
                class_prob_sums += one_hot.T @ probs # (10, E)

                # counts per class
                class_counts += one_hot.sum(dim=0) # (10,)
        
        

        # avoid divide-by-zero; some splits might have rare classes missing
        class_counts = class_counts.clamp_min(1.0)
        class_expert_mean = (class_prob_sums / class_counts.unsqueeze(1)).detach().cpu().numpy()  # (10, E)

        # save into history for plotting later
        history["class_expert_mean"] = class_expert_mean
        history["class_names"] = class_names

        # --- test snapshot of utilization & entropy (SoftMoE only) ---
        # we're measuring the utilization and entropy of the test set
        util_sum_t = torch.zeros(NUM_EXPERTS, device=DEVICE)
        ent_sum_t  = 0.0
        cnt_t = 0
        with torch.no_grad():
            #calculate the utilization and entropy of the test set
            for data, _ in test_loader: #data is (B, 3, 32, 32), _ is (B,)
                data = data.to(DEVICE)
                _, probs, _, _ = model(data, return_gate=True)  # (B, E)
                util_sum_t += probs.sum(dim=0)
                ent_sum_t  += (-(probs * probs.clamp_min(1e-8).log()).sum(dim=1)).sum().item()
                cnt_t      += probs.size(0)

        util_test = (util_sum_t / cnt_t).detach().cpu().numpy()     # shape (E,)
        H_test    = ent_sum_t / cnt_t

        # --- load snapshot ---
        if FF_LAYER == "SparseMoE":
            #calculate the load of the test set
            load_sum_t = torch.zeros(NUM_EXPERTS, device=DEVICE)
            cnt_t_load = 0
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(DEVICE)
                    _, probs, _, _ = model(data, return_gate=True)
                    load_sum_t += (probs > 0).float().sum(dim=0)
                    cnt_t_load += probs.size(0)
            load_test = (load_sum_t / cnt_t_load).detach().cpu().numpy()
            history["load_test_snapshot"] = load_test

        history["util_test_snapshot"] = util_test
        history["entropy_test_snapshot"] = H_test

        # overwrite metrics.pt with the new keys
        torch.save(history, os.path.join(ckpt_dir, "metrics.pt"))
    ### -- END OF softmoe per-class gating probabilities -- ###

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
    elif FF_LAYER == "SparseMoE":
        expert_hidden = int(HIDDEN_MULT * backbone.output_dim)
        total_width   = NUM_EXPERTS * expert_hidden
        print(f"num_experts {NUM_EXPERTS}")
        print(f"top_k       {SPARSEMOE_K}")
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
