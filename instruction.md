
In MP1.1, you will get familiar with building and evaluating Search Engines. In this assignment, we will practice using the pyserini toolkit for indexing and search.

Please follow https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation for installation instruction on your computer. Then download and extract the experimental code from <code_link> and data from <data_link>. Move the data folder into code folder. 

The ranker will be evaluated using NDCG@10 score on 3 relevance datasets: Cranfield dataset, APNews dataset, and the Faculty dataset

**You only need to modify main.py.** Look for places with "###"'s to change specifications. If the installation is correct, the execution should print out score and results.json should appear in the corresponding dataset folder. If you want to rebuild preprocessed corpus or index, simply delete the corresponding data folder in `indexes/` and `processed_corpus/`

This assignment is meant for getting familiar with the toolkit and allowing you to test out BM25 retriever. Your task is to test out different parameters of BM25 and **report the numerical scores on different datasets and parameters you've tried in a single pdf**. **The assignment is graded by completion**. In the next assignment, you will be able to develop your own ranking model with the toolkit for leaderboard evaluation.