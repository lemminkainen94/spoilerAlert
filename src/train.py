import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from time import time
from torch import nn, optim


def train_epoch(args):
    args.model = args.model.train()
    losses = []
    avg_losses = []
    temp_outputs = []
    temp_targets = []
    avg_aurocs = []
    acc_losses = []
    correct_predictions = 0
    i = 0
    t0 = time()
    for d in args.train_dl:
        input_ids = d["input_ids"].to(args.device)
        targets = d["targets"].to(args.device).view(-1, 1)
        outputs = torch.zeros_like(targets)

        if args.arch == 'LSTM':
            outputs = args.model(input_ids)
        else:
            attention_mask = d["attention_mask"].to(args.device)
            outputs = args.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        preds = torch.zeros_like(outputs)
        ones = torch.ones_like(preds)
        preds = torch.where(outputs < 0, preds, ones)

        loss = args.loss_fn(outputs, targets.float())
        loss = loss / args.acc_steps
        correct_predictions += torch.sum(preds == targets)

        temp_outputs += outputs.cpu().tolist()
        temp_targets += targets.cpu().tolist()

        acc_losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(args.model.parameters(), max_norm=1.0)

        i += 1
        if i % args.acc_steps == 0:
            losses.append(np.mean(acc_losses))
            acc_losses = []
            args.optimizer.step()
            if 'scheduler' in args:
                args.scheduler.step()
            args.optimizer.zero_grad()
        if i % (100 * args.acc_steps) == 0:
            roc = 0
            try:
                roc = roc_auc_score(temp_targets, temp_outputs)
            except ValueError:
                pass
            temp_outputs = []
            temp_targets = []

            avg_aurocs.append(roc)
            avg_losses.append(np.mean(losses[i-100:i]))
            print(i, 'iters, auroc, loss, time : ', avg_aurocs[-1], avg_losses[-1], time()-t0)

    return correct_predictions.double() / args.train_size, np.mean(losses), avg_losses, avg_aurocs


if __name__ == '__main__':
    pass
