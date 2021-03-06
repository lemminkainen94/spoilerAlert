import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def eval_model(args):
    args.model = args.model.eval()
    losses = []
    temp_outputs = []
    temp_targets = []   
    correct_predictions = 0
    with torch.no_grad():
        for d in args.eval_dl:
            input_ids = d["input_ids"].to(args.device)
            targets = d["targets"].to(args.device).view(-1, 1)
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
            preds = torch.where(torch.sigmoid(outputs) < args.threshold, preds, ones)

            loss = args.loss_fn(outputs, targets.float())
            correct_predictions += torch.sum(preds == targets)

            temp_outputs += outputs.cpu().tolist()
            temp_targets += targets.cpu().tolist()

            losses.append(loss.item())

    roc = 0
    try:
        roc = roc_auc_score(temp_targets, temp_outputs)
    except ValueError:
        pass

    return correct_predictions.double() / args.eval_size, np.mean(losses), roc


def get_predictions(args):
    args.model = args.model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in args.test_dl:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(args.device)
            attention_mask = d["attention_mask"].to(args.device)
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


if __name__ == '__main__':
    pass