from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.nn import functional as F
from functools import partial
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool
from transformers import AutoTokenizer, AutoModel
from prettytable import PrettyTable
import configparser
import time
import matplotlib.pyplot as plt
import multiprocessing as mp

pd.options.display.max_columns = 100

model_strans = None
tokerizer = None
model = None


def prep_data():
    df_user_info = pd.read_pickle('movielens_100k.pkl')
    return df_user_info


def initialize_models():
    model_strans = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    return model_strans, tokenizer, model


def initialize_models_():
    global model_strans, tokenizer, model
    model_strans, tokenizer, model = initialize_models()


def compare_float_dicts(dict1, dict2, rel_tol=1e-9, abs_tol=0.0):
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        if not torch.allclose(dict1[key], dict2[key], rel_tol, abs_tol):
            return False

    return True

def stub(sentences):
    return generate_sentence_embeddings_explicitly(sentences.tolist(), 'cpu')

def dummy(i):
    pass

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Get token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_sentence_embeddings_explicitly(sentences, device='cpu'):
    model.to(device)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2.0, dim=1)
    return sentence_embeddings

def run(df_user_info, config):
    dict_titlEmb = {}
    dict_titlEmb_ = {}

    model_strans, tokenizer, model = initialize_models()
    # Test 1: Run the sentences serially through sentence transformer and direct embedding calculation and compare
    # run times for different num_threads and batch sizes
    if config['exp1']['active'] == 'True':
        table = PrettyTable()
        table.field_names = ["Batch Size\Execution Time", "Sentence Transformer", "Direct Calculation"]
        # warm up so the model is available in memory
        model.to('cpu')
        model_strans.encode("random string")
        # read batch size config. eval converts string representation into array
        batch_size_cfg = eval(config['exp1']['batch_size'])
        beg = batch_size_cfg[0]
        end = batch_size_cfg[1]
        step = batch_size_cfg[2]
        for batch_size in range(beg, end, step):
            movie_titles = df_user_info["movie_title"].unique()[0:batch_size]
            start = time.time()
            # Encode movie titles using sentence transformer
            for movie_title in movie_titles:
                dict_titlEmb[movie_title] = torch.tensor(model_strans.encode(movie_title))
            t1 = time.time() - start
            print("--- Runtime using sentence transformers %s seconds ---" % t1)

            start = time.time()
            # Now calculate the embeddings directly using the same transformer model and pooling technique
            for movie_title in movie_titles:
               dict_titlEmb_[movie_title]  = generate_sentence_embeddings_explicitly([movie_title])
            t2 = time.time() - start
            print("--- Runtime using direct approach %s seconds ---" % t2)
            table.add_row([("%s" % batch_size), ("%.2f" % t1), ("%.2f" % t2)])
            # Run time on CPU for 600 movie titles is 5.7 sec using sentencetransformers and 3.67 sec using direct approach
            # validate the two dictionaries are nearly identical
            are_close = compare_float_dicts(dict_titlEmb, dict_titlEmb_)
            # assert are_close == True
        print(table)
    if config['exp2']['active'] == 'True':
        # batch execution using direct approach on specified device
        device = 'cpu'
        _device = config['exp2']['device'].lower()
        # default is CPU
        if _device == 'gpu':
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                device = 'cuda'
            else:
                print('CUDA requested,but not available. Reverting to CPU')
        model.to(device)
        table = PrettyTable()
        table.field_names = ["Batch Size\Execution Time", "Direct Calculation: %s" % _device]
        batch_size_cfg = eval(config['exp2']['batch_size'])
        beg = batch_size_cfg[0]
        end = batch_size_cfg[1]
        step = batch_size_cfg[2]
        # warm up on a random string so model is already loaded on the device
        embds = generate_sentence_embeddings_explicitly(['random string'], device)
        # array to store batch size for plotting purposes
        batch_sizes = []
        exec_times = []
        for batch_size in range(beg, end, step):
            # direct approach
            movie_titles = df_user_info["movie_title"].unique()[0:batch_size]
            batch_sizes.append(len(movie_titles)) # because for last batch, size may be < batch_size in the for loop
            start = time.time()
            embds = generate_sentence_embeddings_explicitly(movie_titles.tolist(), device)
            t1 = time.time() - start
            start = time.time()
            exec_times.append(t1)
            # Add to dictionary for comparison with serial approach
            for index, movie_title in enumerate(movie_titles):
                dict_titlEmb_[movie_title] = embds[index]
            print("--- Runtime using direct approach [batch = %d] on %s: %.2f seconds ---" % (batch_size, device, t1))
            # exec_times.append(t1)
            table.add_row([("%s" % batch_size), ("%.2f" % t1)])
        # interestingly, with the batch execution approach, there are more numerical differences between the two
        # dictionaries. For a match, the relative and absolute tolerances have to be increased
        # are_close = compare_float_dicts(dict_titlEmb, dict_titlEmb_, 1e-5, 1e-6)

        print(table)
        plt.plot(batch_sizes[:-1], exec_times[:-1], label=device)
        plt.legend(title='execution times against batch size for CPU/GPU')
        plt.show()

    if config['exp3']['active'] == 'True':
        num_proc = (int)(config['exp3']['num_proc'])

        device = 'cpu'
        chunks = []
        # import os
        # os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # use all movies
        movie_titles = df_user_info["movie_title"].unique()
        # Iterate and slice the list
        chunk_size = len(movie_titles)//num_proc
        batch_size = len(movie_titles)
        for i in range(0, len(movie_titles), chunk_size):
            chunks.append(movie_titles[i:i + chunk_size])
        results_ = []
        generate_sentence_embeddings_explicitly_ = partial(generate_sentence_embeddings_explicitly, device='cpu')

        pool = Pool(initializer = initialize_models_, processes=num_proc)
        # calling a function on the process pool ensures that the initializer is executed.. that way
        # we aren't measuring the time required to load the model in each process
        pool.map(dummy, [i for i in range(num_proc)])
        start = time.time()
        # Below doesn't work.. because lambdas are anonymous and can't be found by the loader
        # results_ = pool.map(lambda x: generate_sentence_embeddings_explicitly_(x.tolist()), chunks)
        # results_ = pool.map(generate_sentence_embeddings_explicitly_, chunks)
        results_ = pool.map(stub, chunks)
        # flatten results
        results = [x for xs in results_ for x in xs]
        t1 = time.time() - start
        print("--- Runtime using direct approach [batch = %d, num_proc = %d] on %s: %.2f seconds ---" %
              (batch_size, num_proc, device, t1))



if __name__ == "__main__":
    mp.set_start_method('spawn')
    config = configparser.ConfigParser()
    config.read('parallelization_experiments.ini')
    # default num_threads on my 20 core machine is 10
    default_num_threads = torch.get_num_threads()
    num_threads = (int)(config['DEFAULT']['num_threads'])
    # Important note: for some values of num_threads (eg., 4 on my machine), use of multi-processing results in a
    # crash.. unclear why. Issue seems to be quite complicated, as indicated here: https://github.com/pytorch/pytorch/issues/17199
    torch.set_num_threads(num_threads)

    df_user_info = prep_data()
    dict_titlEmb = run(df_user_info, config)



