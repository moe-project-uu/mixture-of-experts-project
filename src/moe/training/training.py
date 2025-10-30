import os
import torch
from tqdm import tqdm
from moe.utils.losses import softmoe_load_balance

def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct, targets.size(0)

def training_loop(
    train_loader, 
    val_loader,
    num_epochs, 
    model, 
    optimizer, 
    criterion,
    scheduler,
    softmoe_load_balance_required: bool,
    FF_layer_type: str,
    experts: int,
    DEVICE: str,
    ckpt_model_path = "",
):      
    # --- gating metrics history (SoftMoE only) ---
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        # SoftMoE-specific:
        "util_per_epoch": [],   # utilization per epoch.. list of np arrays shape (E,)
        "entropy_per_epoch": [] # entropy per epoch.. list of floats
    }
    
    pin_memory = (DEVICE == "cuda") 
    
    best_train_acc, best_val_acc = 0.0, 0.0   # Ma, Ga (Ga = best val acc)
    best_train_epoch, best_val_epoch = None, None   # ETT(Ma), ETT(Ga)

    for epoch in range(1, num_epochs + 1):
        # reset per-epoch accumulators for gating stats
        if FF_layer_type == "SoftMoE":
            util_sum = torch.zeros(experts, device=DEVICE) #utilization sum
            ent_sum = 0.0 #entropy sum
            count_samples = 0 
        
        # train
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for data, targets in pbar:
            data, targets = data.to(DEVICE, non_blocking=pin_memory), targets.to(DEVICE, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            ##################
            if FF_layer_type == "Dense":
                logits = model(data, return_gate=False)
            elif FF_layer_type == "SoftMoE":
                logits, probs, _, aux_loss = model(data, return_gate=True) #sel_idx, aux_loss both set to none for now
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
            
            ### -- ADD RELEVANT AUXILIARY LOSS TERMS HERE FOR LOAD BALANCING -- ###
            if FF_layer_type == "SoftMoE" and softmoe_load_balance_required:
                loss = criterion(logits, targets) + softmoe_load_balance(probs, experts, coef=softmoe_load_balance_required)
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
        if FF_layer_type == "SoftMoE" and count_samples > 0:
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
                if FF_layer_type == "Dense":
                    logits = model(data, return_gate=False)
                elif FF_layer_type == "SoftMoE":
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

        print(f"Epoch {epoch:03d}/{num_epochs} | "
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
            print(f"Saved checkpoint: {FF_layer_type} val_acc={best_val_acc*100:.2f}%")

        # step scheduler
        scheduler.step()

    return history, (best_train_acc, best_train_epoch), (best_val_acc, best_val_epoch)