import os, os.path as osp
import json

import pandas as pd
import polars as pl

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from tqdm import tqdm, trange
import igraph as ig
import numpy as np

ROOT = '/fsx/user/HLLM'
NAMES = dict(
    mer = 'merrec_2000',
    pix = 'Pixel8M',
    nerd = 'eb_nerd_512',
    rand = 'Rand',
)
FNAME_PREFIX = os.path.splitext(osp.basename(__file__))[0]
GRAPHS = dict()

def _item_vocab_path(graph_fname: str) -> str:
    base, _ = osp.splitext(graph_fname)
    return base + ".item_vocab.json"

def load_graph(key):
    global GRAPHS
    id2token = {k: [] for k in ['user_id', 'item_id']}
    if key == 'rand':
        return ig.Graph.Erdos_Renyi(n = 100, p = 0.25)
    name = NAMES[key]
    if key in GRAPHS:
        print(f'Using loaded item-item graph of {name}')
    else:
        config = dict()
        config['data_path'] = osp.join(ROOT, 'dataset')  # replace as needed
        config['info_path'] = osp.join(ROOT, 'information')  # replace as needed
        config['random_sample'] = True  # when the selected window is shorter than window_len, we pad with random samples or the 0th sample
        config['sample_last_only'] = False  # we only use the last window for training
        '''Only change the parameters above'''
        if name == 'merrec_2000':
            config['max_user_seq_len'] = 2000 # (qrz): omitting users with extremely long sequences when building the item-item graph
            config['dataset'] = 'merrec_2000'
            config['pred_len'] = 1  # number of items to predict during training
            config['eval_pred_len'] = 1  # number of items to predict during evaluation
            config['min_seq_len'] = 400  # we remove users that engages with fewer than 50 items
            config['category_by'] = 'event'
            config['eval_num_cats'] = 6
            config['train_test_gap'] = 0  # gap between the train split end and the test split start
            config['max_item_list_len'] = 400  # number of items in context
            config['text_keys'] = ['category_name', 'brand_name']
            config['window_len'] = config['pred_len'] + config['max_item_list_len']
        elif name == 'Pixel8M':
            config['max_user_seq_len'] = 200 # (qrz): omitting users with extremely long sequences when building the item-item graph
            config['dataset'] = 'Pixel8M'
            config['pred_len'] = 8  # number of items to predict during training
            config['eval_pred_len'] = 8  # number of items to predict during evaluation
            config['min_seq_len'] = 50  # we remove users that engages with fewer than 50 items
            config['category_by'] = 'item'
            config['eval_num_cats'] = 8
            config['train_test_gap'] = 0  # gap between the train split end and the test split start
            config['max_item_list_len'] = 50  # number of items in context
            config['text_keys'] = ['title', 'tag', 'description']
            config['window_len'] = config['pred_len'] + config['max_item_list_len']
        elif name == 'eb_nerd_512':
            config['max_user_seq_len'] = 2000 # (qrz): omitting users with extremely long sequences when building the item-item graph
            config['dataset'] = 'eb_nerd_512'
            config['pred_len'] = 8  # number of items to predict during training
            config['eval_pred_len'] = 8  # number of items to predict during evaluation
            config['min_seq_len'] = 100  # we remove users that engages with fewer than 50 items
            config['category_by'] = 'item'
            config['eval_num_cats'] = 8
            config['train_test_gap'] = 0  # gap between the train split end and the test split start
            config['max_item_list_len'] = 100  # number of items in context
            config['text_keys'] = ['title', 'tag', 'description']
            config['window_len'] = config['pred_len'] + config['max_item_list_len']
        else:
            raise NotImplementedError
        graph_fname = osp.join(ROOT, f"{name}~max{config['max_user_seq_len']}seq{config['max_item_list_len']}pred{config['pred_len']}~edgelist082225_val.txt")
        GRAPH_FORMAT = 'edgelist'
        vocab_path = _item_vocab_path(graph_fname)
        if osp.exists(graph_fname):
            print(f'Loading preprocessed graph {graph_fname}')
            GRAPHS[key] = ig.Graph.Read(graph_fname, format = GRAPH_FORMAT)
            GRAPHS[key].to_undirected(mode = 'collapse')

            if osp.exists(vocab_path):
                with open(vocab_path, 'r') as f:
                    id2token['item_id'] = json.load(f)
            else:
                id2token['item_id'] = sorted(list(set(pl.Series(
                    GRAPHS[key].vs['name']).to_list())))
        else:
            print(f'Preprocessing {name} into item-item graph')

            interact_feat_path = os.path.join(config['data_path'], f'{config["dataset"]}.parquet')
            if not os.path.isfile(interact_feat_path):
                raise ValueError(f'File {interact_feat_path} not exist.')
            dtype_list = {'item_id': str, 'user_id': str, 'timestamp': int}
            name_list = ['item_id', 'user_id', 'timestamp']
            if config['category_by'] == 'event' and config['eval_num_cats'] > 1:
                print('Category as defined per event. "event_id" column fetched...')
                dtype_list['event_id'] = int
                name_list.append('event_id')
            
            interact_df = pl.read_parquet(
                interact_feat_path, columns=name_list
            )
            
            # remove all users with fewer than min_seq_len or eval_pred_len * 2 interactions
            filter_min_len = config['eval_pred_len'] * 2
            if config['min_seq_len'] is not None:
                filter_min_len = max(config['min_seq_len'], filter_min_len)
            interact_df = interact_df.with_columns(pl.col("item_id").list.len().alias("num_interacts"))
            interact_df = interact_df.filter(pl.col("num_interacts") > filter_min_len)
            interact_df = interact_df.drop("num_interacts")

            id2token['user_id'] = ['[PAD]'] + pl.Series(
                interact_df.select(pl.col('user_id'))).to_list()
            id2token['item_id'] = ['[PAD]'] + sorted(list(set(pl.Series(
                interact_df.select(pl.col('item_id').list.explode())).to_list())))
            token_id = {t: i + 1 for i, t in enumerate(id2token['user_id'][1:])}
            interact_df = interact_df.with_columns(
                pl.col('user_id').replace_strict(token_id, default=-1).alias('user_id'))

            with open(vocab_path, 'w') as f:
                json.dump(id2token['item_id'], f)
            
            user_num = len(id2token['user_id'])
            item_num = len(id2token['item_id'])
            print(f"{user_num = } {item_num = }")
            interact_num = interact_df.select(pl.col('item_id').list.len().sum()).item()  # total number of interactions

            user_seq = [[]] + pl.Series(interact_df.select('item_id')).to_list()
            token_id = {t: i for i, t in enumerate(id2token['item_id'])}
            _get = token_id.__getitem__
            user_seq = [list(map(_get, sublist)) for sublist in user_seq]

            fname = f'{FNAME_PREFIX}~{name}~ulen.pdf'
            if not osp.exists(fname):
                print('Plotting ulen histogram', flush = True)
                plt.figure()
                sns.histplot(x = [len(user_seq[uid]) - config['eval_pred_len'] * 2 - config['train_test_gap'] for uid in range(user_num)])
                plt.title('user seq len')
                plt.savefig(fname)
                plt.close()
                # breakpoint()#########

            edges = set()
            for uid in trange(user_num, desc = 'Building item-item graph'):
                train_seq_len = len(user_seq[uid]) - config['eval_pred_len'] - config['train_test_gap']
                if train_seq_len > 1:
                    edges.update(itertools.combinations(set(user_seq[uid][max(0, train_seq_len - config['max_user_seq_len']): train_seq_len]), 2))
            print(f'num unique edges: {len(edges)}', flush = True)
            edges = list(edges)
            print(f'Converting to iGraph', flush = True)
            GRAPHS[key] = ig.Graph(edges = edges)
            print(f'Writing graph to {graph_fname}', flush = True)
            GRAPHS[key].write(graph_fname, format = GRAPH_FORMAT)

    if key == 'nerd':
        # Build category_list aligned with id2token['item_id']
        info_path = os.path.join(config['info_path'], f'{config["dataset"]}.parquet')
        df_info = pd.read_parquet(info_path, columns=['item_id', 'category'])

        # Keep only items we know about
        vocab = id2token['item_id']
        vocab_set = set(vocab)
        df_info = df_info[df_info['item_id'].isin(vocab_set)].copy()

        # If there are duplicates per item_id, keep the first occurrence
        if df_info.duplicated('item_id').any():
            df_info = df_info.drop_duplicates('item_id', keep='first')

        # Map item_id -> category
        item_to_category = df_info.set_index('item_id')['category'].to_dict()

        # Create list in the exact vocab order; PAD index gets None
        category_list = [-1] + [item_to_category.get(it, -1) for it in vocab[1:]]
    else:
        category_list = None
    return GRAPHS[key], id2token['item_id'], category_list


def _prepare_initial_membership(category_list, graph):
    """
    Convert category_list (strings/ints/None) into a valid membership vector of non-negative ints,
    length == graph.vcount(). Use 0 as the fallback/unknown cluster. Distinct categories map to 1..K.
    """
    if category_list is None:
        return None

    n = graph.vcount()

    # Pad/truncate to match the graph's vertex count
    if len(category_list) < n:
        category_list = list(category_list) + [None] * (n - len(category_list))
    elif len(category_list) > n:
        category_list = list(category_list[:n])

    # If it's already ints (or None), just coerce; else map categories to ints
    if all((c is None) or isinstance(c, (int, np.integer)) for c in category_list):
        memb = [0 if (c is None or (isinstance(c, float) and pd.isna(c))) else int(c)
                for c in category_list]
        # ensure non-negative
        memb = [c if c >= 0 else 0 for c in memb]
        return memb

    # Map arbitrary labels (e.g., strings) -> 1..K, keep 0 for unknown/None
    cat_to_id, next_id = {}, 1
    memb = [0] * n
    for i, c in enumerate(category_list):
        if c is None or (isinstance(c, float) and pd.isna(c)):
            memb[i] = 0
        else:
            k = c
            if k not in cat_to_id:
                cat_to_id[k] = next_id
                next_id += 1
            memb[i] = cat_to_id[k]
    return memb


def cluster(key, graph, obj = 'modularity', resol = 1.0, id2token = None, n_iterations = 5, category_list = None, **kwargs): # kwargs: beta, n_iterations
    name = NAMES[key]
    fname = f'{FNAME_PREFIX}~{name}~O{obj[0]}R{resol}Iter{n_iterations}~max2k~082225~val.json'
    if osp.exists(fname):
        print(f'Clustering is already completed. If you want to re-run it, please delete {fname}')
        return
    # breakpoint()#######

    if category_list is not None:
        initial_membership = _prepare_initial_membership(category_list, graph)
    else:
        initial_membership = None

    print('Running the clustering algorithm')
    res = graph.community_leiden(
        objective_function = obj,
        resolution = resol,
        n_iterations = n_iterations,
        **kwargs,
    )
    # breakpoint()######
    print(f'Got modularity={res.modularity} for resolution={resol}')
    with open(fname, 'w') as fo:
        json.dump(dict(membs = res.membership, modularity = res.modularity, id2token = id2token), fo)


key = 'nerd'
graph, id2token, category_list = load_graph(key)
for resol in [1.5, 1.6, 1.7, 1.8, 1.65, 1.75]:
    for n_iterations in [-1]:
        cluster(key, graph, resol = resol, id2token = id2token, n_iterations = n_iterations, category_list = category_list)