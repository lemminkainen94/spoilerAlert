import argparse
import pandas as pd
import numpy as np
import random
import transformers
import torch
import torch.nn.functional as F

from collections import defaultdict
from sklearn.metrics import roc_auc_score, classification_report
from time import time
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from src.transformer_model import SpoilerClassifier
from src.lstm_model import LSTMSpoilerClassifier
from src.train import train_epoch
from src.eval import eval_model, get_predictions
from src.data_loader import create_data_loader


def run_epoch(args, epoch, writer):
    print(f'Epoch {epoch + 1}/{args.epochs}')
    print('-' * 10)
    train_acc, train_loss, train_avg_losses, train_auroc = train_epoch(args)
    
    print(f'Train loss {train_loss} accuracy {train_acc}  auroc {np.mean(train_auroc)}')
    writer.add_scalars("train", {'Loss': train_loss, 'Auroc': np.mean(train_auroc)}, epoch)

    val_acc, val_loss, val_auroc = eval_model(args)

    print(f'Val   loss {val_loss} accuracy {val_acc} auroc {val_auroc}')
    print()
    writer.add_scalars("val", {'Loss': train_loss, 'Auroc': np.mean(train_auroc)}, epoch)

    args.history['train_auroc'] += train_auroc
    args.history['train_loss'] += train_avg_losses
    args.history['val_auroc'].append(val_auroc)
    args.history['val_loss'].append(val_loss)

    writer.flush()

    if val_auroc > best_auroc:
        if args.out_model is None:
            args.out_model = 'models/' + args.arch + '_' + str(time()) + '.pt' 
        torch.save(args.model.state_dict(), args.out_model)
        best_auroc = val_auroc


def parse_args():
    parser = argparse.ArgumentParser("SploilerAlert")
    parser.add_argument('data', type=str, help='path to data file')
    parser.add_argument('--tokenizer', type=str, default='bert-base-cased',
                        help='Huggingface Transformers pretrained tokenizer. bert-base-cased by default')
    parser.add_argument('--arch', type=str, default='transformer',
                        help='Choose the model architecture for training and/or eval. Can be:\n' +
                        'transformer: the default transformer model, not trained on spoiler detection\n' +
                        'lstm: the default lstm model, not trained on spoiler detection\n' +
                        'More to follow...\n')
    parser.add_argument('--in_model', default=None, help="Path to the model weights to load for training/eval. \n" +
                        "Must conform with the chosen architecture")
    parser.add_argument('--out_model', default=None, help="Name of the trained model (will be saved with that name). Used for training only")
    parser.add_argument("--max_len", default=128, type=int,
                        help="The maximum total input sequence length after BERT tokenization. Sequences "
                        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help="number of training epochs")
    parser.add_argument('--test',
                        default=False,
                        action='store_true',
                        help="Whether on test mode")
    parser.add_argument('--learning_rate',
                        type = float,
                        default = 0.003,
                        help = "Adam (for LSTM) or AdamW (for transformer-based) learning rate. Default is 0.003.")
    parser.add_argument("--hidden_dim",
                        type = int,
                        default = 32,
                        help = "Number of neurons of the hidden layer. Default is 32.") 
    parser.add_argument("--lstm_layers",
                        type = int,
                        default = 2,
                        help = "Number of LSTM layers")              
    parser.add_argument("--batch_size",
                        type = int,
                        default = 32,
                        help = "Batch size")
    parser.add_argument("--final_dropout",
                        type = float,
                        default = 0.3,
                        help = "Dropout before the final linear layer")

    return parser.parse_args()


def main():
    args = parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_pickle(args.data)

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Loading Tokenizer
    args.tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    args.model = None

    if args.arch == 'transformer':
        args.model_name = 'bert-base-cased'
        args.model = SpoilerClassifier(args)
    elif args.arch == 'LSTM':
        args.model = LSTMSpoilerClassifier(args)

    if args.in_model:
        args.model.load_state_dict(torch.load(args.in_model))
    args.model.to(args.device)

    writer = SummaryWriter()

    ### Model Evaluation
    if args.test:
        class_names = ['no_spoilers', 'has_spoilers']
        args.test_dl = create_data_loader(df, args)        
        y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(args)
        print(roc_auc_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=class_names))
        print(y_test.shape, y_pred_probs.flatten().shape)
        writer.add_pr_curve(class_names[1], y_test, y_pred_probs.flatten())
    ### Model Training
    else:
        # final layers fine-tuning
        """for param in model.network.parameters():
            param.requires_grad = False
        
        model.network.final_layer.weight.requires_grad = True
        model.network.final_layer.bias.requires_grad = True
        model.network.pred_final_layer.weight.requires_grad = True
        model.network.pred_final_layer.bias.requires_grad = True"""
        df_train, df_eval = train_test_split(df, test_size=0.2, random_state=args.seed)
        args.train_dl = create_data_loader(df_train)
        args.eval_dl = create_data_loader(df_eval)

        args.train_steps = len(train_dl)
        args.eval_steps = len(eval_dl)

        args.loss_fn = nn.BCEWithLogitsLoss().to(args.device)

        if args.arch == 'transformer':
            args.optimizer = AdamW(args.model.parameters(), lr=args.lr, correct_bias=False)

            args.scheduler = get_linear_schedule_with_warmup(
                args.optimizer,
                num_warmup_steps=0,
                num_training_steps=args.total_steps
            )
        elif args.arch == 'LSTM':
            args.optimizer = optim.Adam(args.model.parameters(), lr=args.lr)

        args.history = defaultdict(list)
        best_auroc = 0
        for epoch in range(args.epochs):
            run_epoch()

    writer.close()


if __name__ == "__main__":
    main()