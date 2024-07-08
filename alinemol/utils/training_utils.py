import numpy as np
import torch
from alinemol.utils import Meter
from alinemol.utils import predict


def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    """
    Run a training epoch over the entire training set.

    Args:
        args (dict): A dictionary of arguments.
        epoch (int): The current epoch.
        model (nn.Module): The model to train.
        data_loader (DataLoader): The DataLoader object for the training set.
        loss_criterion (nn.Module): The loss criterion to use.
        optimizer (Optimizer): The optimizer to use.
    """
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to(args["device"]), masks.to(args["device"])
        logits = predict(args, model, bg)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)
        if batch_id % args["print_every"] == 0:
            print(
                "epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}".format(
                    epoch + 1, args["num_epochs"], batch_id + 1, len(data_loader), loss.item()
                )
            )
    train_score = np.mean(train_meter.compute_metric(args["metric"]))
    print(
        "epoch {:d}/{:d}, training {} {:.4f}".format(
            epoch + 1, args["num_epochs"], args["metric"], train_score
        )
    )


def run_an_eval_epoch(args, model, data_loader, test=False):
    """
    Run an evaluation epoch over the entire validation or test set.

    Args:
        args (dict): A dictionary of arguments.
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The DataLoader object for the validation or test set.
        test (bool): Whether to evaluate on the test set.

    Returns:
        float: The evaluation score.
    """
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args["device"])
            logits = predict(args, model, bg)
            eval_meter.update(logits, labels, masks)

    if test:
        evals = []
        evals.append(np.mean(eval_meter.compute_metric("accuracy_score")))
        evals.append(np.mean(eval_meter.compute_metric("roc_auc_score")))
        evals.append(np.mean(eval_meter.compute_metric("pr_auc_score")))
        return evals
    else:
        return np.mean(eval_meter.compute_metric(args["metric"]))
