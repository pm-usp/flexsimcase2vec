from importlib.metadata import version

def is_dated_gensim_version():
    return version("gensim") < '4.0.0'

import gensim
import random
import itertools
import numpy as np
import pandas as pd
#from .check_gensim import is_dated_gensim_version
from math import ceil

class FlexSimCase2vec:

    def __init__(self, event_log: pd.DataFrame, trace_key: str = 'case',
                 sim_cols: list = ['variant'], sim_cols_weights: list = [1],
                 dimensions: int = 128, walk_length: int = 30, num_walks: int = 10, 
                 workers: int = 1, quiet: bool = False, seed: int = None):
        """
        Initiates the FlexSimCase2vec object, generating walks using pre-defined neighbors.

        :param graph: event log
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param seed: Seed for the random number generator.

        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        
        """

        self.event_log = event_log
        self.trace_key = trace_key
        self.sim_cols = sim_cols
        self.sim_cols_weights = sim_cols_weights
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.quiet = quiet
        
        self.u_traces = event_log[trace_key].unique()


        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.walks = self._generate_trace_walks()

    def _generate_trace_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        #Check if weights were determined and, if so, if their length follows sim_cols
        if len(self.sim_cols_weights) != len(self.sim_cols):
            #If no weigth was determined, define weigth as 1 for all elements in sim_cols
            if len(self.sim_cols) > 1 and len(self.sim_cols_weights) == 1 and self.sim_cols_weights[0] == 1:
                self.sim_cols_weights = [1] * len(self.sim_cols)
            else:
                raise Exception("Size of sim_cols_weights different than size of sim_cols")

        walks = []

        print('Trace identifier:', self.trace_key)
        print('Similarities identifiers:', self.sim_cols)
        print("Similarities identifiers' weigths:", self.sim_cols_weights)
        print("Dimensions:", self.dimensions)
        print("Walk length:", self.walk_length)
        print("#Walks:", self.num_walks)

        for trace_id in self.u_traces:
            neighbors = []
            for i, sim_col in enumerate(self.sim_cols):
                if sim_col not in self.event_log.columns:
                    raise Exception(f"Error! {sim_col} not found!")
                trace_sim_col_values = self.event_log[self.event_log[self.trace_key] == trace_id][sim_col].unique()
                #weights can be add by adding the similar cases found using this sim_col for [WEIGHT] times
                #neighbors are traces that share a characteristic value with the trace under analysis, excluding the trace itself
                neighbors += self.sim_cols_weights[i] * list(self.event_log[(self.event_log[sim_col].isin(trace_sim_col_values)) & (self.event_log[self.trace_key] != trace_id)][self.trace_key].unique())
            if len(neighbors) > 0:
                #simulate neighboors randomly visited, considering the trace under analysis is always revisited after visiting a neighboor
                returns2trace = [trace_id] * ceil(self.walk_length/2)
                trace_walks = [random.choices(neighbors, k=ceil(self.walk_length/2)) for _ in range(self.num_walks)]
                #combine the randomly selected neighboors with the returns to the trace under analysis, ajusting the size of the resulting list if necessary
                combined_walks = [list(itertools.chain.from_iterable(zip(returns2trace,walk)))[:self.walk_length] for walk in trace_walks]
        
                walks += combined_walks  # Add the combined walks to the final result
            else:
                walks += [[trace_id] * self.num_walks]

        return walks

    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameters for gensim.models.Word2Vec - do not supply 'size' / 'vector_size' it is
            taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        # Figure out gensim version, naming of output dimensions changed from size to vector_size in v4.0.0
        size = 'size' if is_dated_gensim_version() else 'vector_size'
        if size not in skip_gram_params:
            skip_gram_params[size] = self.dimensions

        if 'sg' not in skip_gram_params:
            skip_gram_params['sg'] = 1

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)