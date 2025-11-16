import os
import os.path as osp
import json
import pandas as pd
import polars as pl
import itertools
from tqdm import tqdm, trange
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
# IMPORTANT: Please set the ROOT path to the directory containing your dataset files or the 'dataset' subdirectory.
ROOT = '/workspace' 

NAMES = dict(
    mer = 'merrec_2000',
    pix = 'Pixel8M',
    nerd = 'eb_nerd_512',
    rand = 'Rand',
)

# Define FNAME_PREFIX safely
try:
    FNAME_PREFIX = os.path.splitext(osp.basename(__file__))[0]
except NameError:
    FNAME_PREFIX = 'recommendation_clustering'

GRAPHS = dict()

# Determine Polars String type for compatibility
try:
    PL_STRING_TYPE = pl.String
except AttributeError:
    try:
        PL_STRING_TYPE = pl.Utf8
    except AttributeError:
        raise ImportError("Could not determine Polars String/Utf8 type.")

def _vocab_path(graph_fname: str, graph_type: str) -> str:
    base, _ = osp.splitext(graph_fname)
    return base + f".{graph_type}_vocab.json"

def normalize_wide_format_with_timestamp(interact_df: pl.DataFrame) -> pl.DataFrame:
    """
    A more direct normalization function for wide-format data that has an associated timestamp list.
    It ensures items within each user's list are sorted chronologically.
    
    1. Explodes user, item, and timestamp lists to create a long format.
    2. Sorts interactions chronologically for each user.
    3. Groups back into a wide format (user_id, [item_ids]).
    """
    print("Starting direct normalization for wide format with timestamps...")
    
    # Ensure required columns exist
    if not all(c in interact_df.columns for c in ['user_id', 'item_id', 'timestamp']):
        raise ValueError("Input DataFrame must contain 'user_id', 'item_id', and 'timestamp' for this normalization.")

    # 1. Explode to long format
    long_df = interact_df.explode(['item_id', 'timestamp'])
    
    # Drop rows where item_id is null, as they are not useful interactions
    long_df = long_df.filter(pl.col('item_id').is_not_null())

    # 2. Sort by user and then by timestamp to establish correct sequence
    long_df = long_df.sort(['user_id', 'timestamp'])

    # 3. Group back into wide format
    # The item IDs will be aggregated into a list, preserving the chronological order from the sort.
    wide_df = long_df.group_by("user_id").agg(pl.col("item_id"))
    
    print(f"Direct normalization complete. Schema: {wide_df.schema}")
    return wide_df

def normalize_interactions_generic(interact_df):
    """
    Handles data normalization when timestamp is not available or not needed,
    focusing on type casting and ensuring wide format.
    """
    print(f"Starting generic normalization. Initial schema: {interact_df.schema}")

    # Determine if data is in wide format (lists of items)
    is_list_type = False
    if "item_id" in interact_df.schema:
        item_dtype = interact_df.schema["item_id"]
        if isinstance(item_dtype, pl.List):
            is_list_type = True

    if is_list_type:
        # If already wide, just ensure correct types
        print("Data is already in wide format. Ensuring correct types.")
        # Apply cast within the list
        interact_df = interact_df.with_columns(
            pl.col('user_id').cast(PL_STRING_TYPE),
            pl.col('item_id').list.eval(pl.element().cast(PL_STRING_TYPE)).alias('item_id')
        )
    else:
        # If in long format, group into wide format
        print("Data is in long format. Casting and regrouping...")
        interact_df = interact_df.with_columns([
            pl.col('user_id').cast(PL_STRING_TYPE),
            pl.col('item_id').cast(PL_STRING_TYPE)
        ])
        interact_df = interact_df.group_by("user_id").agg(pl.col("item_id"))

    # Sort the final dataframe by user_id for deterministic order
    interact_df = interact_df.sort("user_id")
    print(f"Generic normalization complete. Normalized schema: {interact_df.schema}")
    return interact_df


def load_graph(key, graph_type='item'):
    global GRAPHS
    id2token = {k: [] for k in ['user_id', 'item_id']}

    if graph_type not in ['item', 'user']:
        raise ValueError(f"Invalid graph_type: {graph_type}.")

    name = NAMES.get(key)
    if not name:
        if key == 'rand': return ig.Graph.Erdos_Renyi(n = 100, p = 0.25), [], None
        raise ValueError(f"Invalid dataset key: {key}")
        
    graph_key = f"{key}_{graph_type}"

    # Configuration setup
    config = dict()
    config['data_path'] = osp.join(ROOT, 'dataset')
    
    # Dataset specific configurations
    if name == 'merrec_2000':
        config.update(dataset='merrec_2000', max_user_seq_len=2000, pred_len=1, eval_pred_len=1, min_seq_len=400, train_test_gap=0)
    elif name == 'Pixel8M':
        config.update(dataset='Pixel8M', max_user_seq_len=200, pred_len=8, eval_pred_len=8, min_seq_len=50, train_test_gap=0)
    elif name == 'eb_nerd_512':
        config.update(dataset='eb_nerd_512', max_user_seq_len=2000, pred_len=8, eval_pred_len=8, min_seq_len=100, train_test_gap=0)
    else:
        raise NotImplementedError

    # Define graph filename and vocabulary path
    GRAPH_VERSION = "v13_perf_opt" 
    context_len = config.get('max_user_seq_len', 'NA')
    graph_fname = osp.join(ROOT, f"{FNAME_PREFIX}_{name}~{graph_type}_graph~ctx{context_len}~edgelist_{GRAPH_VERSION}.txt")
    
    GRAPH_FORMAT = 'edgelist'
    vocab_path = _vocab_path(graph_fname, graph_type)
    node_id_key = f'{graph_type}_id'

    if graph_key in GRAPHS:
        print(f'Using loaded {graph_type}-{graph_type} graph of {name}')
        # Ensure vocab is loaded when reusing graph
        if not id2token[node_id_key] and osp.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                id2token[node_id_key] = json.load(f)

    elif osp.exists(graph_fname):
        print(f'Loading preprocessed graph {graph_fname}')
        graph = ig.Graph.Read(graph_fname, format=GRAPH_FORMAT)
        graph.to_undirected(mode='collapse')
        GRAPHS[graph_key] = graph
        with open(vocab_path, 'r') as f:
            id2token[node_id_key] = json.load(f)

    else:
        print(f'Preprocessing {name} into {graph_type}-{graph_type} graph')

        interact_feat_path = osp.join(config['data_path'], f'{config["dataset"]}.parquet')
        if not os.path.isfile(interact_feat_path):
            interact_feat_path = osp.join(ROOT, f'{config["dataset"]}.parquet')
            if not osp.isfile(interact_feat_path):
                raise FileNotFoundError(f'Dataset file not found at {interact_feat_path}')

        df_schema = pl.read_parquet_schema(interact_feat_path)
        cols_to_load = ['user_id', 'item_id']
        if 'timestamp' in df_schema:
            cols_to_load.append('timestamp')
        
        interact_df = pl.read_parquet(interact_feat_path, columns=cols_to_load)

        # --- Refactored Normalization ---
        if 'timestamp' in interact_df.columns and isinstance(interact_df.schema['timestamp'], pl.List):
             # Use timestamp-based normalization for datasets like eb_nerd
             interact_df = normalize_wide_format_with_timestamp(interact_df)
        else:
             # Fallback to generic normalization for others like Pixel8M
             # This assumes item order is already correct if timestamps are absent.
             pass # Will be handled during casting step later

        # --- Cast IDs to String AFTER sorting and grouping ---
        interact_df = interact_df.with_columns([
            pl.col('user_id').cast(PL_STRING_TYPE),
            pl.col('item_id').list.eval(pl.element().cast(PL_STRING_TYPE))
        ])

        # --- Filtering ---
        filter_min_len = max(config.get('min_seq_len', 0), config['eval_pred_len'] * 2)
        interact_df = interact_df.filter(pl.col("item_id").list.len() > filter_min_len)
        
        # Final sort for deterministic user order
        interact_df = interact_df.sort("user_id")

        # --- Deterministic Vocabulary Creation ---
        unique_users = interact_df['user_id'].to_list()
        id2token['user_id'] = ['[PAD]'] + unique_users
        
        exploded_items = interact_df.select(pl.col('item_id').explode()).to_series()
        unique_items = sorted(exploded_items.unique().drop_nulls().to_list())
        id2token['item_id'] = ['[PAD]'] + unique_items

        user_num = len(id2token['user_id'])
        item_num = len(id2token['item_id'])
        print(f"{user_num = } {item_num = }")

        with open(vocab_path, 'w') as f:
            json.dump(id2token[node_id_key], f)

        # --- Graph Construction ---
        edges = []
        if graph_type == 'item':
            user_seq_tokens = [[]] + interact_df['item_id'].to_list()
            item_token_id = {t: i for i, t in enumerate(id2token['item_id'])}
            _get = item_token_id.get # Use .get for safety
            
            edges_set = set()
            for uid in trange(1, len(user_seq_tokens), desc = 'Building item-item graph'):
                seq = [_get(token) for token in user_seq_tokens[uid] if _get(token) is not None]
                train_seq_len = len(seq) - config['eval_pred_len'] - config.get('train_test_gap', 0)
                if train_seq_len > 1:
                    start_idx = max(0, train_seq_len - context_len)
                    window_items = set(seq[start_idx: train_seq_len])
                    edges_set.update(itertools.combinations(sorted(list(window_items)), 2))
            
            edges = list(edges_set)
            num_nodes = item_num

        elif graph_type == 'user':
            print("Mapping User IDs...")
            user_mapping_df = pl.DataFrame({
                'user_id': id2token['user_id'][1:],
                'user_id_int': range(1, user_num)
            })
            interact_df = interact_df.join(user_mapping_df, on='user_id', how='left')
            
            print("Applying temporal window...")
            interact_df = interact_df.with_columns(
                (pl.col('item_id').list.len() - config['eval_pred_len'] - config.get('train_test_gap', 0)).alias('train_seq_len')
            )
            interact_df = interact_df.with_columns(
                (pl.when(pl.col('train_seq_len') > context_len)
                 .then(pl.col('train_seq_len') - context_len)
                 .otherwise(0)).alias('offset')
            )
            interact_df = interact_df.with_columns(
                pl.when(pl.col('train_seq_len') > 0)
                .then(pl.col('item_id').list.slice(pl.col('offset'), pl.col('train_seq_len')))
                .otherwise(pl.lit([], dtype=pl.List(PL_STRING_TYPE)))
                .alias('train_items')
            )

            train_interactions_df = interact_df.select(['user_id_int', 'train_items']).explode('train_items')
            train_interactions_df = train_interactions_df.filter(pl.col('train_items').is_not_null())
            
            print(f"Total interactions in training windows (Context Len={context_len}): {len(train_interactions_df)}")

            # --- Memory-Efficient Edge Generation ---
            print("Building user-user graph via memory-efficient item-wise grouping...")

            # Group by item and collect all users who interacted with that item
            user_groups_by_item = train_interactions_df.group_by('train_items').agg(
                pl.col('user_id_int').unique().alias('user_list')
            )

            # Filter out items that were only touched by one user (no edges can be formed)
            user_groups_by_item = user_groups_by_item.filter(pl.col('user_list').list.len() > 1)
            
            # --- PERFORMANCE OPTIMIZATION ---
            # For very popular items, generating all pairs is too slow.
            # We cap the number of users per item by slicing the list, which is safer than sampling.
            MAX_USERS_PER_ITEM = 2000 
            print(f"Capping user lists at {MAX_USERS_PER_ITEM} to manage performance.")
            
            user_groups_by_item = user_groups_by_item.with_columns(
                pl.when(pl.col('user_list').list.len() > MAX_USERS_PER_ITEM)
                .then(pl.col('user_list').list.slice(0, MAX_USERS_PER_ITEM))
                .otherwise(pl.col('user_list'))
                .alias('user_list')
            )

            edges_set = set()
            # Iterate over the much smaller aggregated dataframe to generate pairs
            for user_list in tqdm(user_groups_by_item.select('user_list').to_series(), desc="Generating user pairs"):
                # Sort the list to ensure canonical edge ordering (u1, u2) where u1 < u2
                sorted_users = sorted(user_list)
                # Generate combinations of user pairs
                edges_set.update(itertools.combinations(sorted_users, 2))

            edges = list(edges_set)
            num_nodes = user_num

        print(f'num unique edges: {len(edges)}')
        if not edges:
            print("Warning: No edges were generated. Graph will be empty.")
            GRAPHS[graph_key] = ig.Graph(n=num_nodes, directed=False)
        else:
            print(f'Converting to iGraph (n={num_nodes})')
            GRAPHS[graph_key] = ig.Graph(n=num_nodes, edges=edges, directed=False)
        
        print(f'Writing graph to {graph_fname}')
        GRAPHS[graph_key].write_edgelist(graph_fname)

    category_list = None
    return GRAPHS[graph_key], id2token[node_id_key], category_list

def _prepare_initial_membership(category_list, graph):
    return None 

def cluster(key, graph, graph_type='item', obj = 'modularity', resol = 1.0, id2token = None, n_iterations = 5, category_list = None, **kwargs):
    name = NAMES.get(key)
    if not name: return
        
    GRAPH_VERSION = "v13_perf_opt"
    
    ctx_len_map = {'Pixel8M': 200, 'eb_nerd_512': 2000, 'merrec_2000': 2000}
    ctx_len = ctx_len_map.get(name, 'NA')

    fname = osp.join(ROOT, f'{FNAME_PREFIX}_{name}~{graph_type}_cluster~ctx{ctx_len}~O{obj[0]}R{resol}Iter{n_iterations}_{GRAPH_VERSION}.json')
    
    if osp.exists(fname):
        print(f'Clustering is already completed. Delete {fname} to re-run.')
        return

    if graph.ecount() == 0:
        print("Warning: Graph has no edges. Skipping clustering.")
        return

    initial_membership = _prepare_initial_membership(category_list, graph)

    print(f'Running Leiden clustering (Resolution={resol}, Iterations={n_iterations})')
    
    try:
        # Note: igraph < 0.10 uses resolution, >= 0.10 uses resolution_parameter
        try:
            res = graph.community_leiden(
                objective_function = obj, resolution_parameter = resol, n_iterations = n_iterations,
                initial_membership=initial_membership, **kwargs
            )
        except TypeError: # Fallback for older igraph versions
             res = graph.community_leiden(
                objective_function = obj, resolution = resol, n_iterations = n_iterations,
                initial_membership=initial_membership, **kwargs
            )
        print(f'Got modularity={res.modularity:.4f} for resolution={resol}. Number of clusters: {len(res)}')
        with open(fname, 'w') as fo:
            json.dump(dict(membs = res.membership, modularity = res.modularity, id2token = id2token), fo)
    except Exception as e:
        print(f"Clustering failed with error: {e}")


# Main execution block
if __name__ == '__main__':
    datasets_to_run = ['pix']
    # To run both, change to: graph_types_to_run = ['item', 'user']
    graph_types_to_run = ['user'] 
    resolutions = [1, 1.1, 1.2]
    n_iterations = -1

    for key in datasets_to_run:
        for graph_type in graph_types_to_run:
            print(f"\n{'='*20} Starting process for dataset: {key}, graph_type: {graph_type} {'='*20}")
            try:
                graph, id2token, category_list = load_graph(key, graph_type=graph_type)
                if graph:
                    print(f"Graph loaded/constructed successfully. Nodes: {graph.vcount()}, Edges: {graph.ecount()}")
                    if graph.ecount() > 0:                    
                        for resol in resolutions:
                            print(f"\nClustering with resolution={resol}, n_iterations={n_iterations}")
                            cluster(key, graph, graph_type=graph_type, resol=resol, id2token=id2token, n_iterations=n_iterations)
                    else:
                        print("Graph is empty. Cannot proceed with clustering.")
            except Exception as e:
                import traceback
                print(f"An unexpected error occurred for dataset {key}, graph_type {graph_type}: {e}")
                traceback.print_exc()

