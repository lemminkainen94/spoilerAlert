import numpy as np
import torch


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    aurocs = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            targets = d["targets"].to(device)
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

    return correct_predictions.double() / n_examples, np.mean(losses), np.mean(aurocs)


def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = torch.zeros_like(targets)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            preds = torch.zeros_like(outputs)
            ones = torch.ones_like(preds)
            preds = torch.where(outputs < 0, preds, ones)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values
