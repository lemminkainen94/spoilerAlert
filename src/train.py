import torch
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
        targets = d["targets"].to(args.device)
        outputs = torch.zeros_like(targets)
        if args.arch == 'LSTM':
            outputs = args.model(input_ids)
        else:
            attention_mask = d["attention_mask"].to(args.device)
            outputs = args.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        aurocs.append(roc_auc_score(targets.cpu(), preds.cpu()))
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(args.model.parameters(), max_norm=1.0)
        args.optimizer.step()
        if args.scheduler:
            args.scheduler.step()
        args.optimizer.zero_grad()
        i += 1
        if i % 100 == 0:
            avg_aurocs.append(np.mean(aurocs[i-100:i]))
            avg_losses.append(np.mean(losses[i-100:i]))
            print(i, 'iters, auroc, loss, time : ', avg_aurocs[-1], avg_losses[-1], time()-t0)

    return correct_predictions.double() / args.train_steps, np.mean(losses), avg_losses, avg_aurocs


if __name__ == '__main__':
    pass
