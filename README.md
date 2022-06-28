# SpoilerAlert
(Work In Progress)  
The goal of this project is to explore the possibility of automated spoiler detection in Goodreads book reviews.  
Currently, Goodreads users are expected to tag their own spoilers when writing the reviews. The reality is a lot of users do not.  
Another way of handling spoilers is to use crowdsourcing, but this method is not very reliable either.  
Many users read reviews to find out whether a specific book is worth their while, but they hate running into spoilers revealing major plot points.  
Hence, there is a real demand for a tool which could detect spoilers automatically.  
In this work I used the [dataset](https://github.com/MengtingWan/goodreads).  
I focused on experimenting with different end-to-end deep learning models to predict spoilers in the reviews on both sentence and review levels. I was able to slightly outperform the existing work, without using any feature engineering, by fine-tuning Huggingface's [electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)


##Usage


## Related Work

For the amount of fury and despair spoilers on the Internet bring upon the World, there is surprisingly little research on how to detect them automatically.  
This project owes a great deal to the contributions of [Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://aclanthology.org/P19-1248) (Wan et al., ACL 2019), which include the creation and sharing of a large dataset of Goodreads reviews with annotated spoilers and an exploration of deep learning methods on the dataset. The reported AUC score was 0.919. The approach in the work mentioned includes some feature engineering based on the reviewed books, which would be problematic if the model were to be used in any practical application where only the review, without any additional data, is provided.  
[Spoiler Alert: Using Natural Language Processing to Detect Spoilers in Book Reviews](https://arxiv.org/abs/2102.03882)(Bao et al., 2021)  experimented with LSTM architecture as well as transformer-based language models, without using any handcrafted features, apart from the book title, which is available for every review anyway, as they found it improves the performance.  
Their best performance on the Goodreads dataset was 0.91, pretty close to the previous work, but without any feature engineering.  
However, in the work mentioned above, they found an LSTM model outperforms transformer-based models by a large margin.  
I found this result quite surprising, so I decided to focus on experimenting with different approaches to fine-tuning these large language models to the task and found choosing the right hyperparameters, undersampling the sentences without spoilers to have a 50/50 class split and using the ELECTRA language model can outperform all the existing work with an AUC score of 0.93, without feature engineering.  

## Data


## Models


## Experiments


## Model Deployment With Flask


## Chrome Extension


## In Practice


## Conclusion and Future Work
