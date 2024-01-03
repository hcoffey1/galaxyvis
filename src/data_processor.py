import os
import shlex
import pandas as pd
import time

from astropy.table import Table
from sklearn.cluster import KMeans, MeanShift, HDBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler

from enum import Enum


class Operation(Enum):
    inRange = 0
    le = 1
    leq = 2
    ge = 3
    geq = 4
    eq =54

from astropy.table import Table

def read_fits(fits_path, hdu=1):
    dat = Table.read(fits_path, format='fits', hdu=hdu)
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    if 'MANGAID' in df:
        df['MANGAID'] = df['MANGAID'].str.decode('utf-8')
        df['MANGAID'] = df['MANGAID'].str.strip()
    return df


def clean_data(config):
    #print(config)

    filePath = "./data/" + config["fileName"] + "." + config["type"]

    df = None

    #Read input file
    if config["type"] == "fits":
       df_list = []
       for hdu in config["hdu"]:
           df_list += [read_fits(filePath, hdu)]

       df = pd.concat(df_list, axis=1)
    
    elif config["type"] == "parquet":
        df = pd.read_parquet(filePath)

    for rule in config["filter"]:
        dropping = False
        contains = False
        base = 0

        #Remove instead of select for
        if rule[base] == "DROP":
            dropping = True
            base += 1
            if rule[base] == "NA":
                df = df.dropna(axis=1)
                continue

        #Partial matching 
        if rule[base] == "CONTAINS":
            contains = True
            base += 1

        term = rule[base]
        base += 1

        op = None
        if rule[base] == "IN" and rule[base+1] == "RANGE":
            op = Operation.inRange
            base += 2
        else:
            if rule[base] == "<":
                op = Operation.le
            elif rule[base] == "<=":
                op = Operation.leq
            elif rule[base] == ">":
                op = Operation.ge
            elif rule[base] == ">=":
                op = Operation.geq
            elif rule[base] == "==":
                op = Operation.eq
            else:
                print("Unsupported operation! ", rule)
            base += 1

        applyAll = False 
        if "\"" in term:
            term = term.strip("\"")
        elif term == "ALL":
            applyAll = True

        if applyAll:
            filtered_columns = df.columns 
        elif contains:
            filtered_columns = [col for col in df.columns if term in col]
        else:
            filtered_columns = [col for col in df.columns if term == col]

        if op == Operation.inRange:
            range = rule[base].split(',')

            mask = ((df[filtered_columns] >= float(range[0])) 
                    & (df[filtered_columns] <= float(range[1]))).all(axis=1)

            if dropping:
                df = df[~mask]
            else:
                df = df[mask]
        
        if op == Operation.eq:
            mask = (df[filtered_columns] == float(rule[base])).all(axis=1)
            if dropping:
                df = df[~mask]
            else:
                df = df[mask]

        elif op == Operation.ge:
            if dropping:
                if applyAll:
                    mask = df.map(lambda x: isinstance(x, (int, float)) and x > float(rule[base])).any(axis=1)
                else:
                    mask = (df[filtered_columns] > float(rule[base])).all(axis=1)
            else:
                if applyAll:
                    mask = df.map(lambda x: isinstance(x, (int, float)) and x <= float(rule[base])).any(axis=1)
                else:
                    mask = (df[filtered_columns] <= float(rule[base])).all(axis=1)

            df = df[~mask]

    return df

def select_columns(df, config):
    columns = []

    for rule in config["select"]:
        base = 0
        contains = False

        if rule[base] == "CONTAINS":
            contains = True
            base += 1

        term = rule[base]
        base += 1

        applyAll = False 
        if "\"" in term:
            term = term.strip("\"")
        elif term == "ALL":
            applyAll = True

        if contains:
            columns += ([col.lower() for col in df.columns if term in col ])#Galaxy Zoo]
        else:
            columns += ([col.lower() for col in df.columns if term == col ])#Galaxy Zoo]

    return list(set(columns))

def parse_config(filePath):

    config = {}
    config["fileName"] = (os.path.splitext(filePath.split('/')[-1])[0])

    filterRules = []
    selectRules = []
    metaRules = []

    filtering = False
    selecting = False


    with open(filePath) as f:
        for line in f.readlines():
            # Remove comments by ignoring text after '#'
            line = line.split('#', 1)[0].strip()
            if line:  # Check if the line is not empty after removing comments
                tokens = shlex.split(line, posix=False)

                if tokens[0] == "TYPE":
                    config['type'] = tokens[1]
                elif tokens[0] == "HDU":
                    config['hdu'] = ([int(x) for x in tokens[1].split(',')])
                elif tokens[0] == "SURVEY":
                    config['survey'] = tokens[1]
                elif tokens[0] == "LABEL":
                    config['label'] = tokens[1]
                elif tokens[0] == "META":
                    metaRules.append(tokens[1:])
                elif tokens[0] == "FILTER":
                    filterRules.append(tokens[1:])
                elif tokens[0] == "SELECT":
                    selectRules.append(tokens[1:])

    config["meta"] = metaRules 
    config["filter"] = filterRules
    config["select"] = selectRules 

    return config

def process_data(filePath):
    #Determine data set type and get config
    dataName = os.path.splitext(filePath.split('/')[-1])[0]
    configPath = "./data_config/" + dataName + ".cfg" 
    config = None

    if os.path.isfile(configPath):
        config = (parse_config(configPath))

    if config == None:
        return

	#Clean data
    #print("Cleaning data")
    df = (clean_data(config))

    #print("Selecting data")
    column_choices = (select_columns(df, config))
    
	#Retrieve column list for embedding choices

	#Return column list, cleaned and meta data
    return [config, df, column_choices]


def read_data(directory):

    dataPairs_manga = []
    dataPairs_decals = []
    try:
        # Iterate over files in the directory
        for filename in os.listdir(directory):
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(os.path.join(directory, filename)):
                pair = process_data(os.path.join(directory, filename))
                if pair != None:
                    if pair[0]['survey'] == 'decals':
                        dataPairs_decals += [pair]
                    elif pair[0]['survey'] == 'manga':
                        dataPairs_manga += [pair]
                    else:
                        print("Warning: Unsupported survey ", pair[0]['survey'])
                #read_data(parse_config(os.path.join(directory, filename)))

    except OSError as e:
        print(f"Error reading directory: {e}")

    return dataPairs_manga, dataPairs_decals

#directory_path = './data'

#read_data(directory_path)
def get_numeric_df(df):
    #Remove non-numeric data
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=non_numeric_columns)
    return df

def run_embedding(df, features, embedding_choice, perplexity, tsne_seed):
    PLOT_XY = None
    embed_df = None

    numeric_df = get_numeric_df(df)
    scaler = MaxAbsScaler()
    scaled_df = scaler.fit_transform(numeric_df[features])

    if embedding_choice == "pca":
        embed_df,PLOT_XY = run_pca(scaled_df)

    elif embedding_choice == "tsne":
        embed_df,PLOT_XY = run_tsne(scaled_df, perplexity=perplexity, seed=tsne_seed)

    return embed_df, PLOT_XY

#Embedding algorithms
def run_pca(df):
    print("Running PCA:")
    start_time = time.time()

    col = ['pc1', 'pc2']
    pca = PCA(n_components=2)
    pca_df = pd.DataFrame(pca.fit_transform(df), columns=col)

    end_time = time.time()
    elapsed_time = end_time - start_time 
    print("Time taken : {} s".format(elapsed_time))

    return pca_df.reset_index(drop=True),col

def run_tsne(df, perplexity=50, seed=42):
    print("Running TSNE, perplexity: {}, seed: {}".format(perplexity, seed))
    start_time = time.time()

    col = ['tsne1', 'tsne2']
    tsne_model = None 
    if seed < 0:
        tsne_model = TSNE(n_components=2, perplexity=perplexity)
    else:
        tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=seed)

    tsne_df = pd.DataFrame(tsne_model.fit_transform(df), columns=col)

    end_time = time.time()
    elapsed_time = end_time - start_time 
    print("Time taken : {} s".format(elapsed_time))

    return tsne_df.reset_index(drop=True),col

def run_isomap(df):
    col = ['iso1', 'iso2']
    tsne_model = Isomap(n_components=2)
    tsne_df = pd.DataFrame(tsne_model.fit_transform(df), columns=col)
    return tsne_df.reset_index(drop=True),col

#Clustering algorithms
def run_kmeans(df, k, seed):
    if seed < 0:
        kmeans = KMeans(n_clusters=k, n_init=10)
    else:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)

    kmeans.fit(df)
    return kmeans.labels_.astype(str)

def run_meanshift(df):
    return MeanShift().fit(df).labels_.astype(str)

def run_hdbscan(df):
    return HDBSCAN(min_cluster_size=20).fit(df).labels_.astype(str)

def run_agglomerative(df, k):
    agglocluster = AgglomerativeClustering(n_clusters=k)
    return agglocluster.fit(df).labels_.astype(str)

def run_clustering(df, algo_name, num_k, k_seed):
    clusters = None
    if algo_name == 'kmeans':
        clusters = run_kmeans(df, num_k, k_seed) 
    elif algo_name == 'meanshift':
        clusters = run_meanshift(df)
    elif algo_name == 'hdbscan':
        clusters = run_hdbscan(df)
    elif algo_name == 'agglomerative':
        clusters = run_agglomerative(df, num_k)

    return clusters

def merge_data(data_pairs, key):
    merge_df = data_pairs[0][1]
    selected_features = [[data_pairs[0][0]['label'], data_pairs[0][2]]]
    for pair in data_pairs[1:]:
        merge_df = (merge_df.merge(pair[1], left_on=key, right_on=key))
        selected_features += [[pair[0]['label'], pair[2]]]

    return merge_df, selected_features

def prepare_data(file_path):
    #Read and clean data
    data_pairs_manga, data_pairs_decals = (read_data(file_path))

    #Merge dataframes
    merge_df_manga, selected_features_manga = merge_data(data_pairs_manga, 'MANGAID')
    merge_df_decals, selected_features_decals = merge_data(data_pairs_decals, 'iauname')
    print(merge_df_decals)
    print(selected_features_decals)

    #print(selected_features)
    merge_df_manga.rename(columns={col: col.lower() for col in merge_df_manga.columns}, inplace=True)
    merge_df_decals.rename(columns={col: col.lower() for col in merge_df_decals.columns}, inplace=True)

    #Remove non-numeric data
    numeric_df_manga = get_numeric_df(merge_df_manga)
    numeric_df_decals = get_numeric_df(merge_df_decals)

    #Select features of interest
    selected_manga = []
    for f in selected_features_manga:
        selected_manga += (f[1])

    selected_decals = []
    for f in selected_features_decals:
        selected_decals += (f[1])

    numeric_df_manga = numeric_df_manga[selected_manga]
    numeric_df_decals = numeric_df_decals[selected_decals]

    merge_df_manga = merge_df_manga.reset_index(drop=True)
    numeric_df_manga = numeric_df_manga.reset_index(drop=True)

    merge_df_decals = merge_df_decals.reset_index(drop=True)
    numeric_df_decals = numeric_df_decals.reset_index(drop=True)

    merge_df_manga = pd.concat([numeric_df_manga, merge_df_manga['mangaid']], axis=1, ignore_index=False)
    merge_df_decals = pd.concat([numeric_df_decals, merge_df_decals['iauname']], axis=1, ignore_index=False)

    return [numeric_df_manga, merge_df_manga, selected_features_manga], [numeric_df_decals, merge_df_decals, selected_features_decals]