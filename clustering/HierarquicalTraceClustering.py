import os
import time
import math
import csv
import json
import pickle
import heapq
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm


#-----------------------------#
#        GLOBAL VARS          #
#-----------------------------#

LOG_ID = 'incident'

ATTRIBS = ['category', 'priority', 'variant', #'trace_act',
           'TimeSincePreviousEvent-case_avg', #'TimeSinceCaseStarted-case_avg', 
           'Weekday-case_avg', 'SameDayTime-case_avg', 'Duration']
ATTRIBS_IDS_DIR = '../../../preprocess/results/case_attribs_ids/'
ATTRIBS_IDS_FILES = {attrib: [filename for filename in os.listdir(ATTRIBS_IDS_DIR) if '.csv' in filename and LOG_ID in filename and attrib in filename] for attrib in ATTRIBS}
print(ATTRIBS_IDS_FILES)

APPLY_CONSTRAINTS_VALUES = [False]#[False, True]
CALC_ML_METRICS = False
CONSTRAINTS_ATTRIB_VEC = None #'../../../preprocess/results/constraints/easy_categs.csv'
CONSTRAINTS_COL = None #'label-easy_categs'
RANDOM_ML_VEC = None #'../../../preprocess/results/constraints/random_ml.csv'
RANDOM_ML_COL = None #'random_ml'

RES_DIR = '../results/'
DIR_CLUSTERING_DFS = f'{RES_DIR}clustered_dfs/'
DIR_SUBLOGS = f'{RES_DIR}sublogs/'
METADATA_FILENAME = 'metadata'

INDEX_COL = 'number' #'CaseID'
CLUSTER_COL = 'group'
VEC_SPACE_DIR = None
VECTORS = None
K = 5

N3_KS = [1,100] 

VEC_SPACES_DATA = [
  #{'attrib_vec': '../data/attrib_vecs/log_attrib_exp/log_attrib_exp-not_filtered.csv',
  # 'dist': 'euclidean'},

  #{'attrib_vec': '../../preprocess/results/attrib_vecs/log_attrib_exp/log_attrib_exp-not_filtered-with_constraint_col.csv',
  # 'dist': 'euclidean'},

  #{'attrib_vec': '../../preprocess/results/attrib_vecs/log_attrib_exp/log_attrib_exp_without_priority-not_filtered.csv',
  # 'dist': 'euclidean'},

  #{'attrib_vec': '../../preprocess/results/attrib_vecs/log_attrib_exp/log_attrib_exp_without_priority-not_filtered-with_constraint_col.csv',
  # 'dist': 'euclidean'},
       
  #{'attrib_vec': '../data/attrib_vecs/det_act_act_tfidf_attrib_vec/act_tfidf_attrib_vec_bin-det_act-not_filtered.csv', 
  # 'dist': 'euclidean'},

  #{'attrib_vec': '../../preprocess/results/attrib_vecs/det_act_act_tfidf_attrib_vec/act_tfidf_attrib_vec_bin-det_act-not_filtered-with_constraint_col.csv',
  # 'dist': 'euclidean'},

  #{'attrib_vec': '../../preprocess/results/attrib_vecs/det_act_act_tfidf_attrib_vec/act_tfidf_attrib_vec_bin_without_priority-det_act-not_filtered.csv', 
  # 'dist': 'euclidean'},

  #{'attrib_vec': '../../preprocess/results/attrib_vecs/det_act_act_tfidf_attrib_vec/act_tfidf_attrib_vec_bin_without_priority-det_act-not_filtered-with_constraint_col.csv',
  # 'dist': 'euclidean'},

  #{'seq': '../data/sequences/det_act-not_filtered.csv', 
  # 'attrib_vec': '../data/attrib_vecs/categ_priority/categ_priority-not_filtered.csv', 
  # 'dist': 'seq_attr', 'seq_weight': 1, 'vec_weight': 1,
  # 'unique_seq_d': '../data/dist_dicts/det_act-not_filtered-seq-norm.pickle', 
  # 'unique_vec_d': '../data/dist_dicts/det_act-not_filtered-attrib_vec_categ_priority-norm.pickle'},

  #{'seq': '../data/sequences/det_act-not_filtered.csv', 
  # 'attrib_vec': '../data/attrib_vecs/categ_priority/categ_priority-not_filtered.csv', 
  # 'dist': 'seq_attr', 'seq_weight': 100, 'vec_weight': 1,
  # 'unique_seq_d': '../data/dist_dicts/det_act-not_filtered-seq-norm.pickle', 
  # 'unique_vec_d': '../data/dist_dicts/det_act-not_filtered-attrib_vec_categ_priority-norm.pickle'}, 

  #{'seq': '../data/sequences/det_act-not_filtered.csv', 
  # 'attrib_vec': '../data/attrib_vecs/categ_priority/categ_priority-not_filtered.csv', 
  # 'dist': 'seq_attr', 'seq_weight': 1, 'vec_weight': 0,
  # 'unique_seq_d': '../data/dist_dicts/det_act-not_filtered-seq-norm.pickle', 
  # 'unique_vec_d': '../data/dist_dicts/det_act-not_filtered-attrib_vec_categ_priority-norm.pickle'}, 
  
  #{'seq': '../data/sequences/def_act-filtered_act_count_outliers.csv', 
  # 'attrib_vec': '../data/attrib_vecs/categ_priority/categ_priority-filtered_act_count_outliers.csv', 
  # 'dist': 'seq_attr', 'seq_weight': 1, 'vec_weight': 1
  # 'unique_seq_d': '../data/dist_dicts/det_act-filtered_act_count_outliers-seq-norm.pickle', 
  # 'unique_vec_d': '../data/dist_dicts/det_act-filtered_act_count_outliers-attrib_vec_categ_priority-norm.pickle'}, 
  

  #{'attrib_vec': '../data/attrib_vecs/node2vec/node2vec_8d_P1-graph_attrib_act_variant_8d_edge1.csv', 
  #                         'dist': 'euclidean'}, #the same vector as this graph version with edges 01!!!
]

trace2bpvector = [{'attrib_vec': f"../../../preprocess/Trace2BPVec/results/{filename}", 'dist': 'euclidean'} for filename in os.listdir('../../../preprocess/Trace2BPVec/results/') if 'csv' in filename]
#baseline_one_hot = [{'attrib_vec': f"../../../preprocess/results/one_hot_baseline_vectors/{filename}", 'dist': 'euclidean'} for filename in os.listdir('../../../preprocess/results/one_hot_baseline_vectors/') if 'csv' in filename]
#node2vec_dicts = [{'attrib_vec': f"../../../preprocess/graph_embeddings/results/node2vec/{filename}", 'dist': 'euclidean'} for filename in os.listdir('../../../preprocess/graph_embeddings/results/node2vec/') if 'csv' in filename]
#metapath_dicts = [{'attrib_vec': f"../../graph_embeddings/results/metapath2vec/{filename}", 'dist': 'euclidean'} for filename in os.listdir('../../graph_embeddings/results/metapath2vec/') if 'csv' in filename]
#case2vec_dicts = [{'attrib_vec': f"../../../preprocess/case2vec/results/{filename}", 'dist': 'euclidean'} for filename in os.listdir('../../../preprocess/case2vec/results/') if 'csv' in filename]
#trace2vec_dicts = [{'attrib_vec': f"../../../preprocess/trace2vec/results/{filename}", 'dist': 'euclidean'} for filename in os.listdir('../../../preprocess/trace2vec/results/') if 'csv' in filename]

#VEC_SPACES_DATA = VEC_SPACES_DATA + node2vec_dicts + metapath_dicts
#VEC_SPACES_DATA = VEC_SPACES_DATA + node2vec_dicts
#VEC_SPACES_DATA = VEC_SPACES_DATA + baseline_one_hot #+ case2vec_dicts + trace2vec_dicts 
VEC_SPACES_DATA = VEC_SPACES_DATA + trace2bpvector 
#VEC_SPACES_DATA = VEC_SPACES_DATA + case2vec_dicts 
#VEC_SPACES_DATA = VEC_SPACES_DATA + trace2vec_dicts 


#-----------------------------#
#         FUNCTIONS           #
#-----------------------------#

def cluster(starttime, result_prefix):
  global VECTORS
  VECTORS = get_csv(VEC_SPACE_DIR['attrib_vec'])
  #result = linkage(VECTORS.drop(columns=[INDEX_COL]).values, method='ward') #if the d_matrix isn't used after
  calc_d_matrix()
  time_end_d_matrix = time.time()
  time_d_matrix = time_end_d_matrix - starttime
  print('\nDone! Now clustering...')
  result = linkage(np.concatenate(VECTORS['dists'].values), method='ward')
  time_clust = time.time() - time_end_d_matrix
  print('Done!')
  
  print('\nSaving clustering...')
  VECTORS[CLUSTER_COL] = fcluster(result, K, criterion='maxclust')
  save_csv(pd.DataFrame(VECTORS[[INDEX_COL, CLUSTER_COL]]),f'{DIR_CLUSTERING_DFS}{result_prefix}.csv')
  print('Done!')
  return time_d_matrix, time_clust


def calc_d_matrix():

  def get_d_lines():
    
    def d_line_pairs_euclidean(x):
      ys = VECTORS.drop(columns=INDEX_COL).reset_index().iloc[int(x['index'])+1:].values[:, 1:]
      return np.sqrt(np.sum((ys - x.values[1:]) ** 2, axis=1))

    def d_line_pairs_seqvec():
      global VECTORS
      print("\nCalculating 'seqvec' distance..")
      attrib_vec_cols = [col for col in VECTORS.columns if col != INDEX_COL]
      VECTORS['str_vec'] = VECTORS.apply(lambda x: np.array2string(x[attrib_vec_cols].values), axis=1)
      VECTORS = VECTORS.merge(get_csv(VEC_SPACE_DIR['seq'])).drop(columns=attrib_vec_cols+[INDEX_COL]).values
      
      print('Loading precalculated distances for unique sequences and unique vectors...')
      unique_vec_d = get_pickle(VEC_SPACE_DIR['unique_vec_d'])
      unique_seq_d = get_pickle(VEC_SPACE_DIR['unique_seq_d'])
      print('Done!')
      vec_w = VEC_SPACE_DIR['vec_weight']
      seq_w = VEC_SPACE_DIR['seq_weight']

      print('Defining distaces per line...')
      n = len(VECTORS)
      dists = []
      for i in tqdm(range(n)):
        line = VECTORS[0]
        VECTORS = VECTORS[1:]
        
        line_dists = []
        for line2 in VECTORS:
          distance_vec = 0 if line[0] == line2[0] else unique_vec_d[f'{line[0]}--{line2[0]}'] if f'{line[0]}--{line2[0]}' in unique_vec_d else unique_vec_d[f'{line2[0]}--{line[0]}'] #considering that 0 is the seq string column ("str_vec")
          distance_seq = 0 if line[1] == line2[1] else unique_seq_d[f'{line[1]}--{line2[1]}'] if f'{line[1]}--{line2[1]}' in unique_seq_d else unique_seq_d[f'{line2[1]}--{line[1]}'] #considering that 1 is the seq string column ("trace")
          line_dists.append((seq_w*distance_seq + vec_w*distance_vec)/(seq_w+vec_w))
        dists.append(line_dists)
      
      return np.array(dists)

    #main of function get_d_lines
    if VEC_SPACE_DIR['dist'] == 'euclidean':
      print("\nCalculating 'euclidean' distance..")
      return VECTORS.drop(columns=INDEX_COL).reset_index().apply(d_line_pairs_euclidean, axis=1)
    return d_line_pairs_seqvec()

  def apply_ml_constraints():
    if APPLY_CONSTRAINTS:
      attrib_vec = pd.read_csv(CONSTRAINTS_ATTRIB_VEC).merge(VECTORS)[[INDEX_COL, CONSTRAINTS_COL]]
      ml_indexes = attrib_vec.reset_index()[attrib_vec[CONSTRAINTS_COL]==1]['index'].values
      for i, index1 in enumerate(ml_indexes):
        for index2 in ml_indexes[i+1:]:
          VECTORS['dists'].iloc[index1][index2-index1-1] = 0
  
  global VECTORS
  index_col = VECTORS[INDEX_COL]
  dists = get_d_lines()
  VECTORS = pd.DataFrame({INDEX_COL: index_col, 'dists': dists})
  apply_ml_constraints()


def postprocess_result(result_prefix, n3_ks, easycategs_ml_df, easycategs_ml_col, random_ml_df, random_ml_col):

  def calc_N2_N3(ks):

    attribs_n3_errors = {attrib: np.zeros(len(ks)) for attrib in ATTRIBS_IDS}
    n3_easycategs_ml_errors = np.zeros(len(ks))
    n3_random_ml_errors = np.zeros(len(ks))
    n3_errors = np.zeros(len(ks))

    #just to be sure that the labels are linked to the right case id
    attribs_ids_dfs = {attrib: VECTORS[[INDEX_COL]].merge(ATTRIBS_IDS[attrib]) for attrib in ATTRIBS_IDS}
    if CALC_ML_METRICS:
      df_easycategs_ml = VECTORS[[INDEX_COL]].merge(easycategs_ml_df) 
      df_random_ml = VECTORS[[INDEX_COL]].merge(random_ml_df)

    intra_sum = 0
    extra_sum = 0

    n = len(VECTORS)

    for i in tqdm(range(n)):

      dists = get_dists(i)

      attribs_masks = {attrib: np.array(attribs_ids_dfs[attrib][f'{attrib}_id'] == attribs_ids_dfs[attrib].iloc[i][f'{attrib}_id']) for attrib in ATTRIBS_IDS}
      attribs_masks = {attrib: np.delete(attribs_masks[attrib],i) for attrib in ATTRIBS_IDS} 

      cluster_mask = np.array(VECTORS[CLUSTER_COL] == VECTORS.iloc[i][CLUSTER_COL])
      cluster_mask = np.delete(cluster_mask,i)

      if CALC_ML_METRICS:
        easycategs_ml_mask = np.array(df_easycategs_ml[easycategs_ml_col] == 1)
        easycategs_ml_mask = np.delete(easycategs_ml_mask,i)

        random_ml_mask = np.array(df_random_ml[random_ml_col] == 1)
        random_ml_mask = np.delete(random_ml_mask,i)

      sorted_indexes = np.argsort(dists)
      for ki,k in enumerate(ks):
        knn_mask = np.zeros_like(dists, dtype=bool)
        knn_mask[sorted_indexes[:k]] = True
        if CALC_ML_METRICS:
          if df_easycategs_ml.iloc[i][easycategs_ml_col] == 1:
            n3_easycategs_ml_errors[ki] += k - sum(easycategs_ml_mask[knn_mask])
          if df_random_ml.iloc[i][random_ml_col] == 1:
            n3_random_ml_errors[ki] += k - sum(random_ml_mask[knn_mask])
        n3_errors[ki] += k - sum(cluster_mask[knn_mask])
        for attrib in ATTRIBS_IDS:
          attribs_n3_errors[attrib][ki] += k - sum(attribs_masks[attrib][knn_mask])

      intra_dists = dists[cluster_mask]
      extra_dists = dists[~cluster_mask]
      intra_min = 0 #for the cases where the datapoint is the only one in the group!
      if len(intra_dists) > 0:
        intra_min = np.min(intra_dists)
      extra_min = np.min(extra_dists)
      intra_sum += intra_min
      extra_sum += extra_min

    attribs_n3_values = {attrib: np.array(attribs_n3_errors[attrib])/ks/n for attrib in ATTRIBS_IDS}
    
    n3_easycategs_ml_values = [None]*len(ks)
    n3_random_ml_values = [None]*len(ks)
    if CALC_ML_METRICS:
      n3_easycategs_ml_values = np.array(n3_easycategs_ml_errors)/ks/easycategs_ml_df[easycategs_ml_col].sum()
      n3_random_ml_values = np.array(n3_random_ml_errors)/ks/random_ml_df[random_ml_col].sum()
    
    n3_values = np.array(n3_errors)/ks/n


    intraextra = intra_sum / extra_sum
    n2 = 1 - 1 / (1 + intraextra)

    return attribs_n3_values, n3_easycategs_ml_values, n3_random_ml_values, n3_values, n2

  def get_dists(i):
    dists = np.array(VECTORS.iloc[i]['dists'])
    lines_dists = VECTORS['dists'].values
    previous_lines_dists = [lines_dists[j][i-j-1] for j in range(len(lines_dists[:i]))]
    if len(previous_lines_dists) > 0: 
      dists = np.concatenate([previous_lines_dists, dists])
    return dists


  def clustering_entropy():

    def cluster_entropy(group):
      p_ij = group.shape[0]/total_points
      return  p_ij * math.log(p_ij)
    total_points = VECTORS.shape[0]
    entropy = VECTORS.groupby(CLUSTER_COL).apply(cluster_entropy).sum()
    if entropy == 0:
      return entropy
    return -1 * entropy / math.log(K)


  def calc_CI_categ_constr():

    def cluster_entropy(group):
      p_ij = group.shape[0]/total_class_points
      return  p_ij * math.log(p_ij)

    if CALC_ML_METRICS:
      class_df = VECTORS[[INDEX_COL,CLUSTER_COL]].merge(get_csv(CONSTRAINTS_ATTRIB_VEC))
      count_per_cluster = class_df.groupby(CLUSTER_COL)['label-easy_categs'].sum().values
      class_df = class_df[class_df['label-easy_categs']==1]
      total_class_points = class_df.shape[0]
      entropy = class_df.groupby(CLUSTER_COL).apply(cluster_entropy).sum()
      if entropy == 0:
        return entropy, count_per_cluster
      return -1 * entropy / math.log(K), count_per_cluster
    return None, [None]*K

  def calc_CI_attribs():

    def cluster_entropy(group):
      p_ij = group.shape[0]/total_class_points
      return  p_ij * math.log(p_ij)

    len_dataset = VECTORS.shape[0]
    attribs_ids_dfs = {attrib: VECTORS[[INDEX_COL,CLUSTER_COL]].merge(ATTRIBS_IDS[attrib]) for attrib in ATTRIBS_IDS}
    attribs_CIs = {attrib: 0 for attrib in ATTRIBS_IDS}

    for attrib in ATTRIBS_IDS:
      #print(f'\n\n\n{attrib}')
      attrib_values = attribs_ids_dfs[attrib][f'{attrib}_id'].value_counts().to_dict()
      #print(f'Value counts: \n{attrib_values}')
      for value in  attrib_values:
        #print(f'\n===\nValue: {value}\n===\n')
        attrib_value_df = attribs_ids_dfs[attrib][attribs_ids_dfs[attrib][f'{attrib}_id']==value]
        #print(f'DF: {attrib_value_df.head()}')
        total_class_points = attrib_value_df.shape[0]
        #print(f'Total {value} in dataset:',total_class_points)
        attrib_value_entropy = attrib_value_df.groupby(CLUSTER_COL).apply(cluster_entropy).sum()
        if attrib_value_entropy != 0:
          #print(f'Accumulated entropy for value {value} along the clusters: {attrib_value_entropy}')
          attrib_value_entropy = -1 * attrib_value_entropy  / math.log(K) * total_class_points / len_dataset 
          #print(f'Final entropy for value {value} (divided by log{K} clusters and ponderada pela fraçao de {value} em {attrib}: {attrib_value_entropy}')
          attribs_CIs[attrib] += attrib_value_entropy
    print(f'CIs: {attribs_CIs}')
    return attribs_CIs


  print('Calculating metrics...')
  attribs_N3_values, N3_easycategs_ml_values, N3_random_ml_values, N3_values, N2 = calc_N2_N3(n3_ks)
  attribs_CI_values = calc_CI_attribs()
  CI = clustering_entropy()
  CI_categs_constr, count_per_cluster = calc_CI_categ_constr()
  elems_qtt = VECTORS[CLUSTER_COL].value_counts().sort_index()
  print('Done!')

  return attribs_N3_values, N3_easycategs_ml_values, N3_random_ml_values, N3_values, N2, attribs_CI_values, CI, CI_categs_constr, count_per_cluster, elems_qtt.values


def get_result_id():
  result_id = VEC_SPACE_DIR['attrib_vec'].split('/')[-1].replace('.csv','')
  if 'seq' in VEC_SPACE_DIR:
    result_id = f"seqattrib_{VEC_SPACE_DIR['seq_weight']}s_{VEC_SPACE_DIR['vec_weight']}a-det_act-{result_id.split('-')[-1]}"
    #all the results involving sequences are on top of det_act
    #taking from the attrib_vec filename only the information of filters
  return f"hierarq_ward_categs_constr_{APPLY_CONSTRAINTS}-{result_id}"


def get_pickle(filename):
  return pickle.load(open(filename,'rb'))


def get_csv(filename):
  df = pd.read_csv(filename)
  if df.columns[0] == 'Unnamed: 0':
    df = pd.read_csv(filename, index_col=0)
  return df


def save_csv(df, filename):
  df.to_csv(filename, index=False)


def create_metadata_file():
    filename = f'{RES_DIR}{METADATA_FILENAME}_{time.strftime("%Y%m%d%H%M%S")}.csv'
    attribs_N3_cols = ','.join([f'N3_{col}_{str(k)}' for col in ATTRIBS_IDS_FILES.keys() for k in N3_KS ])
    N3_ec_cols = ','.join([f'N3_easycategs_ml_{str(k)}' for k in N3_KS])
    N3_r_cols = ','.join([f'N3_random_ml_{str(k)}' for k in N3_KS])
    N3_cols = ','.join([f'N3_{str(k)}' for k in N3_KS])
    N3_all_cols = ','.join([N3_ec_cols, N3_r_cols, N3_cols])
    attribs_CI_cols = ','.join([f'CI_{col}' for col in ATTRIBS_IDS_FILES.keys()])
    header = f'id,time_d_matrix,time_clust,time_total,{attribs_N3_cols},{N3_all_cols},N2,{attribs_CI_cols},CI,CI_categs_constr,cluster,categs_constr_count,elem_qtt'
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(header.split(','))
    return filename


def save_metadata(filename, id, time_d_matrix, time_clust, time_total, attribs_N3_values,
                  N3_easycategs_ml_values, N3_random_ml_values, N3_values, N2, 
                  attribs_CI_values, CI, CI_categs_constr, categs_constr_count, elems_qtt):
  attribs_N3_values = ','.join([','.join([str(n3_val) for n3_val in attribs_N3_values[attrib]]) for attrib in ATTRIBS_IDS])
  N3_easycategs_ml_values = ','.join([str(val) for val in N3_easycategs_ml_values])
  N3_random_ml_values = ','.join([str(val) for val in N3_random_ml_values])
  N3_values = ','.join([str(val) for val in N3_values])
  N3_all_values = ','.join([N3_easycategs_ml_values, N3_random_ml_values, N3_values])
  attribs_CI_values = ','.join([str(attribs_CI_values[attrib]) for attrib in ATTRIBS_IDS])
  cluster_independent = f'{id},{time_d_matrix},{time_clust},{time_total},{attribs_N3_values},{N3_all_values},{N2},{attribs_CI_values},{CI},{CI_categs_constr}'
  with open(filename, 'a') as csvfile:
      spamwriter = csv.writer(csvfile)
      for i in range(len(elems_qtt)):
          new = cluster_independent.split(',') + ([i, categs_constr_count[i], elems_qtt[i]])
          spamwriter.writerow(new)


#-----------------------------#
#            MAIN             #
#-----------------------------#

print(f'Quantity of vector spaces being processed: {len(VEC_SPACES_DATA)}')

metadata_filename = create_metadata_file()
ATTRIBS_IDS = {attrib: get_csv(f'{ATTRIBS_IDS_DIR}{ATTRIBS_IDS_FILES[attrib][0]}')[[INDEX_COL, f'{attrib}_id']] for attrib in ATTRIBS_IDS_FILES}

for APPLY_CONSTRAINTS in APPLY_CONSTRAINTS_VALUES:
  print(f'\n\n##########\nApply constrained clustering: {APPLY_CONSTRAINTS}\n##########')
  for i, VEC_SPACE_DIR in enumerate(VEC_SPACES_DATA):
    result_id = get_result_id()
    print(f'\n\n======= File {i+1}: {result_id} =======')
    print(result_id)
    starttime = time.time()

    time_d_matrix, time_clust = cluster(starttime, result_id)

    print('\nPostprocessing result...')
    easycategs_ml_df = random_ml_df = None
    if CALC_ML_METRICS:
      easycategs_ml_df = get_csv(CONSTRAINTS_ATTRIB_VEC)[[INDEX_COL, CONSTRAINTS_COL]]
      random_ml_df = get_csv(RANDOM_ML_VEC)[[INDEX_COL, RANDOM_ML_COL]]
    post_process_results = postprocess_result(result_id, N3_KS, 
                                              easycategs_ml_df, CONSTRAINTS_COL,
                                              random_ml_df, RANDOM_ML_COL,
                                              )

    print('\nSaving metrics in metadata file...')
    save_metadata(metadata_filename, result_id, time_d_matrix, time_clust, time.time()-starttime, *post_process_results)
    print('Done!')