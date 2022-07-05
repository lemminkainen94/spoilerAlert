# SpoilerAlert
(Work In Progress)  
The goal of this project is to explore the possibility of automated spoiler detection in Goodreads book reviews.  
Currently, Goodreads users are expected to tag their own spoilers when writing the reviews. The reality is a lot of users do not.  
Another way of handling spoilers is to use crowdsourcing, but this method is not very reliable either.  
Many users read reviews to find out whether a specific book is worth their while, but they hate running into spoilers revealing major plot points.  
Hence, there is a real demand for a tool which could detect spoilers automatically.  
In this work I used the [dataset](https://github.com/MengtingWan/goodreads).  
I focused on experimenting with different end-to-end deep learning models to predict spoilers in the reviews on both sentence and review levels. I was able to slightly outperform the existing work, without using any feature engineering, by fine-tuning Huggingface's [electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)


## Usage
Overall, this provides a solid base for building a spoiler detection Chrome extension. For now it works pretty poorly, but in the future it might become more useful with more powerful models.    

You can look at the dataset format in the data folder, where I uploaded the file I used for evaluating the models. I used data_prep.ipynb notebook to prepare the data.  

```pip install -r requirements.txt```  
If you want to use a GPU and pytorch doesn't detect it, you might want to run: ```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch```  

#### Training
You can explore the different arguments in ```main.py```. For example, to train an electra-small-discriminator with the default settings (0.3 final dropout, batch size of 32, max input length of 128 tokens) for 5 epochs, without gradient accumulation and random seed fixed to 42, using a dataset in ```data/train.pkl``` you can run:  
```python main.py data/train.pkl --model_name google/electra-small-discriminator --out_model models/electra_ep5.pt```  

#### Evaluation
If you wanted to test the model, obtained by the command above, on ```data/test.pkl```, you could run:  
```python main.py data/test.pkl --in_model models/electra_ep5.pt```

#### Running the Flask server
```python parser.py``` will start up a Flask server that you can send html requests. It loads whatever model you specify in your config file. An example config file:  
```
{
	"arch": "transformer",
	"model_name": "google/electra-small-discriminator",
	"in_model": "models/electra_ep5_non_weighted.pt",
	"max_len": 128,
	"acc_steps": 1,
	"batch_size": 32,
	"final_dropout": 0.3,
	"threshold": 0.12
}
```

#### Chrome extension
Is located in ```app/chrome``` folder of this repo. You can activate it in your Chrome extension manager. Then click on it in your extensions tab. it should dynamically tag some sentences on the viewed webpage. Note that it only works with Goodreads reviews. Sometimes it might lag and you might relload the page and click on the extension icon again to run it.

## Related Work

For the amount of fury and despair spoilers on the Internet bring upon the World, there is surprisingly little research on how to detect them automatically.  
This project owes a great deal to the contributions of [Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://aclanthology.org/P19-1248) (Wan et al., ACL 2019), which include the creation and sharing of a large dataset of Goodreads reviews with annotated spoilers and an exploration of deep learning methods on the dataset. The reported AUC score was 0.919. The approach in the work mentioned includes some feature engineering based on the reviewed books, which would be problematic if the model were to be used in any practical application where only the review, without any additional data, is provided.  
[Spoiler Alert: Using Natural Language Processing to Detect Spoilers in Book Reviews](https://arxiv.org/abs/2102.03882)(Bao et al., 2021)  experimented with LSTM architecture as well as transformer-based language models, without using any handcrafted features, apart from the book title, which is available for every review anyway, as they found it improves the performance.  
Their best performance on the Goodreads dataset was 0.91, pretty close to the previous work, but without any feature engineering.  
However, in the work mentioned above, they found an LSTM model outperforms transformer-based models by a large margin.  
I found this result quite surprising, so I decided to focus on experimenting with different approaches to fine-tuning these large language models to the task and found choosing the right hyperparameters, undersampling the sentences without spoilers to have a 50/50 class split and using the ELECTRA language model can outperform all the existing work with an AUC score of 0.93, without feature engineering.  

## Data

The [dataset](https://github.com/MengtingWan/goodreads) contains, among others, a file with met-data of 2.36M books, as well as 15M Goodreads book reviews, 1.38M of which are annotated with spoilers, by parsing spoiler tags on Goodreads. The data is highly skewed - only about 3% of reviews have spoiler tags, which is probably the main issue as the real number of spoilers is probably much higher, which means the data might not be the best representation of teh true landscape.  
For traning I used the subset extracted by (Bao et al., 2021). It contains 275k reviews with 3.5M sentences.  
In my experiments I used three versions of the dataset:
- Sentence-level with all the sentences, each annotated with 1 if it contains a spoiler and 0 otherwise. Only about 3% of the sentences contain a spoiler.
- Random subset of 270k sentences, obtained by (Bao et al., 2021)
- Subset of the version above, obtained by undersampling the sentences with no reviews, so that the classes are 50/50. This one contains 228k sentences.
- Review-level with all the reviews. I used the text from the whole review as the input and annotated reviews with 1 if they contain at least one sentence marked as a spoiler.  

## Models

For comparison purposes, I transformed the sentence level dataset version into tf-idf vectors and fitted a Naive Bayes classifier on these vectors.  
I experimented with 3 different Huggingface language models:
- [bert-base-cased](https://huggingface.co/bert-base-cased)
- [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
- [electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)  

The main idea behind this choice was to see if language models smaller than BERT can perform as well and whether other pre-training techniques can provide ELECTRA with some edge in this task. Having a smaller and faster would be favourable in case of productionizing.  

In terms of model architecture, it's pretty standard: To handle the spoiler detection task I use each language model to create the input sentence (or the whole review) representation and then apply dropout and feed that to a fully connected layer, which outputs the class logits.  

## Experiments

The NB TF-IDF baseline had 0.68 AUC score when trained on the balanced subset and 0.72 when trained on the full dataset.  
Of the three transformer architectures I was able to achieve the best result of 0.93 AUC with ELECTRA-SMALL-DISCRIMINATOR.   
I found that the dataset used had a fundamental influence on the performance, with the random subset of 270k sentences giving the best result (the same model trained on the full dataset had only 0.75 AUC, while training on the balanced one gave 0.87 AUC, training on the review level 0.71).  
I tried various dropout rate before the final layer, as well as different batch sizes, between 8 and 512. Dropout of 0.3 and batch size of 32 gave the best results.  
As the main problem of the model was the unbalanced dataset I tried applying higher weights to the positive class, but it did not help as it only traded off some precision for a bit of recall.  
Pre-training a masked language model on the reviews did not help either, with the pre-trained ELECTRA model achieving 0.905 AUC.  


## In Practice

I've deployed the model with flask and created a Chrome extension to send current displayed page's html and get it back with spoilers tagged and dynamically modify the html. The results are quite disappointing, given the high AUC score on the dataset, with the sentences tagged as spoilers being picked not much better than at random, which shows the model is very brittle to the inputs received and the dataset may not be representative of the true task.

## Conclusion and Future Work

Although the experiments were quite promising, with the best model receiving 0.93 AUC score, the model doesn't work well in practice. Apart from the potential problems with input format and the dataset used for training, it may just be that the task of spoiler detection is too difficult to handle with no book info injected into the models (apart from the titles). It is indeed difficult, even for a human, to decide whether something is a spoiler without knowing the plot.  
I think it might be worth thinking about the problem in an Information Retrieval context. Given a database of book summaries in a form of a list of key plot points, the model could use IR methods of query-key similarity to check if the given span of text is similar to any of the plot points and then decide if the span contains a spoiler. It might be challenging to obtain such a database, but I believe one could use classic works of literature, which have lots of summaries, analyses and reviews written about them, to experiment with the idea. This is the next thing I'd like to try in the future.
