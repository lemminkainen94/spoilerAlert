import argparse
import pandas as pd
import numpy as np
import transformers
import torch
from collections import defaultdict
from time import time


def run_epoch(args, epoch):
	print(f'Epoch {epoch + 1}/{args.epochs}')
  	print('-' * 10)
	train_acc, train_loss, train_avg_losses, train_auroc = train_epoch(
		args.model,
	    args.train_data_loader,
	    args.loss_fn,
	    args.optimizer,
	    args.device,
	    args.train_len,
	    scheduler=args.scheduler,
	    model_type=args.arch
	)
  	
  	print(f'Train loss {train_loss} accuracy {train_acc}  auroc {np.mean(train_auroc)}')
  	
  	val_acc, val_loss, val_auroc = eval_model(
    	model,
    	val_data_loader,
	    loss_fn,
	    device,
	    args.val_len,
	    model_type=args.arch
  	)

  	print(f'Val   loss {val_loss} accuracy {val_acc} auroc {val_auroc}')
  	print()
	args.history['train_auroc'] += train_auroc
	args.history['train_loss'] += train_avg_losses
	args.history['val_auroc'].append(val_auroc)
	args.history['val_loss'].append(val_loss)
	if val_auroc > best_auroc:
    	torch.save(model.state_dict(), args.out_model)
    	best_auroc = val_auroc


def parse_args():
    parser = argparse.ArgumentParser("SploilerAlert")
    parser.add_argument('data', type=str, help='path to data file')
    parser.add_argument('tokenizer', type=str, default='bert-base-cased'
    					help='Huggingface Transformers pretrained tokenizer. bert-base-cased by default')
	parser.add_argument('arch', type=str, default='transformer',
	                    help='Choose the model architecture for training and/or eval. Can be:\n' +
	                    'transformer: the default transformer model, not trained on spoiler detection\n' +
	                    'lstm: the default lstm model, not trained on spoiler detection\n' +
	                    'More to follow...\n')
    parser.add_argument('--in_model', default=None, help="Path to the model weights to load for training/eval. \n" +
    					"Must conform with the chosen architecture")
    parser.add_argument('--out_model', default=None, help="Name of the trained model (will be saved with that name). Used for training only")
    parser.add_argument("--max_seq_length", default=128, type=int,
                		help="The maximum total input sequence length after BERT tokenization. Sequences "
                    	"longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--seed',
                    	type=int,
                    	default=42,
                    	help="random seed for initialization")
    parser.add_argument('--epochs',
                    	type=int,
                    	default=2,
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


def main():
	args = parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	df = pd.read_pickle(args.data)

	# Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Loading Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    args.model = None

    if args.arch == 'transformer':
        args.model = SpoilerClassifier(args)
    elif args.arch == 'LSTM':
        args.model = LSTMSpoilerClassifier(args)
    
    if args.in_model:
    	args.model.load_state_dict(torch.load(args.in_model))
    args.model.to(device)

    ### Model Evaluation
    if args.test:
        y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_dl)
		print(roc_auc_score(y_test, y_pred))
		print(classification_report(y_test, y_pred, target_names=class_names))
    ### Model Training
    else:
        # final layers fine-tuning
        """for param in model.network.parameters():
            param.requires_grad = False
        
        model.network.final_layer.weight.requires_grad = True
        model.network.final_layer.bias.requires_grad = True
        model.network.pred_final_layer.weight.requires_grad = True
        model.network.pred_final_layer.bias.requires_grad = True"""

        total_steps = len(train_data_loader) * args.epochs
		loss_fn = nn.BCEWithLogitsLoss().to(device)

		if args.arch == 'transformer':
			optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

			scheduler = get_linear_schedule_with_warmup(
			 	optimizer,
			 	num_warmup_steps=0,
			 	num_training_steps=total_steps
			)
		elif args.arch == 'LSTM':
			optimizer = optim.Adam(model.parameters(), lr=0.003)

		history = defaultdict(list)
		best_auroc = 0
		for epoch in range(EPOCHS):
			run_epoch()

if __name__ == "__main__":
    main()