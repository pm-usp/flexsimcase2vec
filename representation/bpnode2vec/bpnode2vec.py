import os
import warnings
import pickle
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

class BPNode2vec:
    def __init__(
        self,
        log_df: pd.DataFrame,
        log_id: str = 'log',
        case_col_key: str = 'case:concept:name',
        activity_col_key: str = None,
        timestamp_col_key: str = None,
        variant_nodes: bool = False,
        variant_col_key: str = None,
        case_attr_nodes: list = [],
        event_nodes: bool = False,
        event_id_col_key: str = None,
        event_position_nodes: bool = False,
        event_position_col_key: str = None,
        event_attr_nodes: list = [],
        dimensions: int = 128,
        walk_length: int = 30,
        p: float = 3.0,
        window_size: int = 3,
        resulting_graph_dir: str = "./",
        force_graph_creation: bool = False,
        resultind_embeddings_dir: str = './',
        force_emb_model_training: bool = False,
        only_training_set: bool = False,
        training_set_perc: float = 0.7,
        seed: int = 1,
        verbose: bool = True
    ):
        """
        Initialize BPNode2vec class.

        The following attribute columns can be specified in case_attr_nodes or event_attr_nodes even if they are not present in the input log_df:
            - 'duration': computed as the time between first and last event in a case
            - 'time_since_previous_event': computed as the time difference between an event and the one before it in the same case
            - 'weekday' and 'same_day_time': derived from the timestamp column

        Parameters:
            log_df (pd.DataFrame): Event log as a DataFrame.
            log_id (str): Identifier for naming saved files.
            case_col_key (str): Column name representing case IDs.
            activity_col_key (str): Column name representing activity names.
            timestamp_col_key (str): Column with event timestamps.
            variant_nodes (bool): Whether to include variant nodes.
            variant_col_key (str): Column with variant IDs.
            case_attr_nodes (list): Case-level attributes to include as nodes.
            event_nodes (bool): Whether to include event nodes.
            event_id_col_key (str): Column name for event ID. Uses row index if None.
            event_position_nodes (bool): Whether to include position-based event nodes.
            event_position_col_key (str): Column with event position (i.e., the position of the event within its case).
            event_attr_nodes (list): Event attributes to link to event nodes. If event_nodes is False, these attributes are aggregated at case level using the methods defined in event_attr_case_agg_mode.
            dimensions (int): Embedding size.
            walk_length (int): Length of random walks.
            p (float): Return parameter for node2vec.
            window_size (int): Context window size for Word2Vec.
            resulting_graph_dir (str): Directory to store graph files.
            force_graph_creation (bool): Whether to always recreate the graph.
            resultind_embeddings_dir (str): Directory to store embeddings.
            force_emb_model_training (bool): Force retrain of embedding model.
            only_training_set (bool): Restrict to training cases.
            training_set_perc (float): Proportion of data used for training.
            seed (int): Random seed.
            
        """

        self.log_df = log_df.copy()
        self.log_id = log_id
        self.case_col_key = case_col_key
        self.activity_col_key = activity_col_key
        self.timestamp_col_key = timestamp_col_key
        self.variant_nodes = variant_nodes
        self.variant_col_key = variant_col_key
        self.case_attr_nodes = case_attr_nodes
        self.event_nodes = event_nodes
        self.event_id_col_key = event_id_col_key
        self.event_position_nodes = event_position_nodes
        self.event_position_col_key = event_position_col_key
        self.event_attr_nodes = event_attr_nodes
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.p = p
        self.window_size = window_size
        self.resulting_graph_dir = resulting_graph_dir
        self.force_graph_creation = force_graph_creation
        self.resultind_embeddings_dir = resultind_embeddings_dir
        self.force_emb_model_training = force_emb_model_training
        self.only_training_set = only_training_set
        self.training_set_perc = training_set_perc
        self.seed = seed
        self.verbose = verbose
        
        self.graph = None
        self.log = self.log_df.copy()
        self.model = None
        self.case_embeddings = None

        if not self.verbose:
            warnings.filterwarnings('ignore')

        self._preprocess_log()
        self._create_graph()

    def _warn(self, msg):
        if self.verbose:
            warnings.warn(msg)

    def _info(self, msg):
        if self.verbose:
            print(f"[{self.__class__.__name__}] {msg}")

    def _preprocess_log(self):

        def _define_cases2use():
            if self.only_training_set:
                cases = sorted(self.log[self.case_col_key].unique())
                train_cases = cases[:int(len(cases) * self.training_set_perc)]
                self.log = self.log[self.log[self.case_col_key].isin(train_cases)]

        def _validate_main_log_info():
            #Validate case column
            if self.case_col_key not in self.log.columns:
                raise ValueError(f"Required case column '{col}' not found in the log.")

            #Validate activity information, either through activity column or variant column
            if self.variant_nodes:
                if self.variant_col_key not in self.log.columns:
                    if self.activity_col_key and self.activity_col_key in self.log.columns:
                        self.variant_nodes = False
                        self._warn(f"Variant column {self.variant_col_key} not found. Graph will be generated with ACTIVITY NODES instead.")
                    else:
                        raise ValueError(f"Variant column {self.variant_col_key} not found and fallback (activity column {self.activity_col_key}) also not found! Please provide one of them correctly.")
                else:
                    if self.activity_col_key and self.activity_col_key in self.log.columns:
                        self._warn(f"Option for variant nodes is True and its column {self.variant_col_key} is valid in the log, so the activity nodes will NOT be added to the graph - activity column {self.activity_col_key} will be ignore, even though it is valid.")
                        if self.event_nodes:
                            self._warn(f"ATTENTION: Please be aware that event nodes will not be linked to any activity notion, since the variant information is linked to the case node!")
            else:
                if not self.activity_col_key or self.activity_col_key not in self.log.columns:
                    raise ValueError(f"Required activity column not found in the log. Provide a valid column for activity or opt for variant nodes (variant_nodes=True) and provide a valid column for variant information via variant_col_key.")

            for col, name in [(self.case_col_key, 'case'), (self.activity_col_key, 'activity')]:
                if col not in self.log.columns:
                    raise ValueError(f"Required {name} column '{col}' not found in the log.")

            #Validate timestamp column if indicated and use it to sort the log; ignore it and warn about it if not validated
            if self.timestamp_col_key and self.timestamp_col_key in self.log.columns:
                self.log = self.log.sort_values(by=[self.case_col_key, self.timestamp_col_key]).reset_index(drop=True)
            else:
                self.timestamp_col_key = None
                self._warn(f"Timestamp column '{self.timestamp_col_key}' not found. It will not be used.")
                
        def _check_paths():
            for path, label in [(self.resulting_graph_dir, 'resulting_graph_dir'), (self.resultind_embeddings_dir, 'resultind_embeddings_dir')]:
                if not os.path.exists(path):
                    raise ValueError(f"The specified directory for '{label}' does not exist: {path}")

        def _check_special_columns():
            if self.event_nodes and (self.event_id_col_key is None or self.event_id_col_key not in self.log.columns):
                self._warn("event_id column not found. Using row index as fallback.")
                self.log['event_id'] = self.log.index
                self.event_id_col_key = 'event_id'
            if self.event_position_nodes and not self.event_nodes:
               self._warn("Event position node is True but event node is False, so event position will not be added to the graph!")
               self.event_position_nodes = False
            elif self.event_position_nodes and (self.event_position_col_key is None or self.event_position_col_key not in self.log.columns):
                self._warn("Event position column not found. Generating sequential positions by case.")
                self.log['event_position'] = self.log.groupby(self.case_col_key).cumcount() + 1
                self.event_position_col_key = 'event_position'

        def _warn_missing_attributes():
            for node_type, attrs in [('case', self.case_attr_nodes), ('event', self.event_attr_nodes)]:
                for attr in attrs:
                    if attr not in self.log.columns:
                        attrs.remove(attr)
                        self._warn(f"{node_type.capitalize()} attribute '{attr}' not found in log and will be ignored.")

        _define_cases2use()
        _validate_main_log_info()
        _check_paths()
        _check_special_columns()
        _warn_missing_attributes()

    def _create_graph(self):

        def _build_graph_filename():
            attr_str = '-'.join(self.case_attr_nodes) if self.case_attr_nodes else 'False'
            event_attrs = '-'.join(self.event_attr_nodes) if self.event_attr_nodes else 'False'
            self.graph_filename = f"{self.log_id}--evnt{self.event_nodes}--evntpos{self.event_position_nodes}--attribs${attr_str}--eventattribs${event_attrs}--variant{self.variant_nodes}--trainingset{self.only_training_set}.pickle"
            return os.path.join(self.resulting_graph_dir, self.graph_filename)

        def _load_existing_graph(path):
            self._info("Loading existing graph...")
            with open(path, 'rb') as f:
                self.graph = pickle.load(f)
            return self.graph

        def _generate_edges():
            
            def add_edges(row):
                case_node = f"case#{row[self.case_col_key]}"
                activity_node = f"activity#{row[self.activity_col_key]}"
                if self.event_nodes:
                    event_node = f"event#{row[self.event_id_col_key]}"
                    self.graph.add_edge(event_node, activity_node, weight=1)
                    self.graph.add_edge(event_node, case_node, weight=1)
                    if self.event_position_nodes and self.event_position_col_key in row:
                        position_node = f"eventpos#{row[self.event_position_col_key]}"
                        self.graph.add_edge(event_node, position_node, weight=1)
                    for attr in self.event_attr_nodes:
                        if attr in row:
                            self.graph.add_edge(event_node, f"{attr}#{row[attr]}", weight=1)
                else:
                    if self.variant_nodes:
                        self.graph.add_edge(case_node, f"variant#{row[self.variant_col_key]}", weight=1)
                    else:
                        self.graph.add_edge(case_node, activity_node, weight=1)
                    for attr in self.event_attr_nodes:
                        if attr in row:
                            self.graph.add_edge(case_node, f"{attr}#{row[attr]}", weight=1)

                for attr in self.case_attr_nodes:
                    if attr in row:
                        self.graph.add_edge(case_node, f"{attr}#{row[attr]}", weight=1)

            self._info("Generating graph...")             
            self.graph = nx.Graph()            
            self.log.apply(add_edges, axis=1)
            with open(graph_path, 'wb') as f:
                pickle.dump(self.graph, f)

        def _summarize_graph():
            self._info(f"\n==========\nGraph saved with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            from collections import Counter
            type_counts = Counter(str(n).split('#')[0] for n in self.graph.nodes)
            for node_type, count in type_counts.items():
                self._info(f"- {node_type} nodes: {count}")
            self._info('\n==========')

        graph_path = _build_graph_filename()
        if os.path.exists(graph_path) and not self.force_graph_creation:
            _load_existing_graph(graph_path)
        else:
            _generate_edges()
        _summarize_graph()



    def fit(self):
        self._info("Starting model fitting process...")
        model_name = f"node2vec_{self.dimensions}d_P{self.p}-{self.graph_filename.replace('.pickle','')}"
        model_path = os.path.join(self.resultind_embeddings_dir, "models", f"{model_name}.model")
        embeddings_path = os.path.join(self.resultind_embeddings_dir, f"{model_name}.csv")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if os.path.exists(model_path) and not self.force_emb_model_training:
            self._info("Loading existing Node2Vec model...")
            self.model = Word2Vec.load(model_path)
        else:
            self._info("Training Node2Vec model... (progress below from node2vec)")
            node2vec = Node2Vec(self.graph, dimensions=self.dimensions, walk_length=self.walk_length, p=self.p, seed=self.seed)
            self.model = node2vec.fit(window=self.window_size)
            self._info(f"Saving model to: {model_path}")
            self.model.save(model_path)

        self._info("Creating case embedding DataFrame...")
        case_nodes = [n for n in self.graph.nodes if str(n).startswith("case#")]
        vectors = [self.model.wv[n] for n in case_nodes]
        case_ids = [n.replace("case#", "") for n in case_nodes]
        self.case_embeddings = pd.DataFrame(vectors)
        self.case_embeddings[self.case_col_key] = case_ids

        self._info(f"Saving case embeddings to: {embeddings_path}")
        self.case_embeddings.to_csv(embeddings_path, index=False)

        return self.model, self.case_embeddings
