import torch
from torch import nn, optim


def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    n_examples,
    scheduler=None,
    model_type='transformer'
):
    model = model.train()
    losses = []
    avg_losses = []
    aurocs = []
    avg_aurocs = []
    correct_predictions = 0
    i = 0
    t0 = time()
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        targets = d["targets"].to(device)
        outputs = torch.zeros_like(targets)
        if model_type == 'LSTM':
            outputs = model(input_ids)
        else:
            attention_mask = d["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        aurocs.append(roc_auc_score(targets.cpu(), preds.cpu()))
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        i += 1
        if i % 100 == 0:
            avg_aurocs.append(np.mean(aurocs[i-100:i]))
            avg_losses.append(np.mean(losses[i-100:i]))
            print(i, 'iters, auroc, loss, time : ', avg_aurocs[-1], avg_losses[-1], time()-t0)

    return correct_predictions.double() / n_examples, np.mean(losses), avg_losses, avg_aurocs
