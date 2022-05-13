import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from time import time
from torch import nn, optim


def train_epoch(args):
    args.model = args.model.train()
    losses = []
    avg_losses = []
    aurocs = []
    avg_aurocs = []
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
        correct_predictions += torch.sum(preds == targets)

        roc = 0
        try:
            roc = roc_auc_score(targets.cpu(), outputs.cpu().detach().numpy())
        except ValueError:
            pass
        aurocs.append(roc)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(args.model.parameters(), max_norm=1.0)
        args.optimizer.step()
        if 'scheduler' in args:
            args.scheduler.step()
        args.optimizer.zero_grad()
        i += 1
        if i % 100 == 0:
            avg_aurocs.append(np.mean(aurocs[i-100:i]))
            avg_losses.append(np.mean(losses[i-100:i]))
            print(i, 'iters, auroc, loss, time : ', avg_aurocs[-1], avg_losses[-1], time()-t0)

    return correct_predictions.double() / args.train_size, np.mean(losses), avg_losses, avg_aurocs


if __name__ == '__main__':
    pass
