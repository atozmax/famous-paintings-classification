import copy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve_path
from src.data import NumericImageFolder, make_loaders
from src.model import create_model, pick_device


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    truths = []
    preds = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        truths.append(labels.cpu())
        preds.append(logits.argmax(dim=1).cpu())
    y_true = torch.cat(truths).numpy()
    y_pred = torch.cat(preds).numpy()
    return {
        "loss": total_loss / total,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    config = load_config("train")
    seed_everything(config["seed"])
    device = pick_device()
    data_dir = resolve_path(config["data_dir"])
    checkpoint = resolve_path(config["checkpoint"])
    output = resolve_path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    base = NumericImageFolder(data_dir)
    train_idx, val_idx = train_test_split(
        np.arange(len(base)),
        test_size=config["val_fraction"],
        random_state=config["seed"],
        stratify=base.targets,
    )
    train_loader, val_loader = make_loaders(
        base,
        train_idx,
        val_idx,
        config["batch_size"],
        config["num_workers"],
    )
    print(f"device: {device}")
    print(f"train: {len(train_idx)}   validation: {len(val_idx)}   classes: {len(base.classes)}")

    model = create_model(
        len(base.classes),
        checkpoint=checkpoint,
        pretrained=bool(config["pretrained"]),
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, config["epochs"] + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        print(
            f"epoch {epoch:03d}/{config['epochs']}  lr {learning_rate:.1e}  "
            f"train acc {train_metrics['accuracy']:.3f} f1 {train_metrics['f1']:.3f}  "
            f"val acc {val_metrics['accuracy']:.3f} f1 {val_metrics['f1']:.3f}",
            flush=True,
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            continue
        epochs_without_improvement += 1
        if epochs_without_improvement % config["lr_patience"] == 0:
            optimizer.param_groups[0]["lr"] *= config["lr_factor"]
            print(f"lr -> {optimizer.param_groups[0]['lr']:.1e}", flush=True)
        if epochs_without_improvement >= config["early_stop_patience"]:
            print(f"stopping at epoch {epoch}", flush=True)
            break

    torch.save(best_state, output)
    print(f"saved {output}  best val accuracy {best_accuracy:.3f}")


if __name__ == "__main__":
    main()
