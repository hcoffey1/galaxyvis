#Galaxy Vis : Interactive Visualization Tool for MaNGA galaxies
#Hayden Coffey

import dash
from dash.dependencies import Input, Output, State, ALL
from dash import ctx
import pandas as pd
import time

from datetime import datetime
import os

import webbrowser

from astropy.table import Table

from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler

from sklearn.cluster import KMeans, MeanShift, HDBSCAN, AgglomerativeClustering

from layout import get_page_layout, get_scatter_fig, get_cluster_scatter_fig, \
    get_cluster_line_fig, get_cluster_bar_fig

from data_processor import read_data
from src.bpt import prepare_bpt_data 

PLOT_XY = []

CURRENT_EMBEDDING = None
CURRENT_CLUSTERING = None

# Define the path to your FITS file
zoo_df17_file_path = "./data/MaNGA_gz-v2_0_1.fits"
firefly_file_path= "./data/manga-firefly-globalprop-v3_1_1-mastar.fits"
dap_all_file_path = "./data/dapall-v3_1_1-3.1.0.fits"
morph_fits = "./data/manga_visual_morpho-2.0.1.fits"

dim_red_df = None

def read_fits(fits_path, hdu=1):
    dat = Table.read(fits_path, format='fits', hdu=hdu)
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    if 'MANGAID' in df:
        df['MANGAID'] = df['MANGAID'].str.decode('utf-8')
        df['MANGAID'] = df['MANGAID'].str.strip()
    return df

def get_numeric_df(df):
    #Remove non-numeric data
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=non_numeric_columns)
    return df

def run_embedding(df, features, embedding_choice, perplexity, tsne_seed):
    PLOT_XY = None
    embed_df = None

    numeric_df = get_numeric_df(df)
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

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

data_pairs = (read_data("./data"))

#Merge dataframes
merge_df = data_pairs[0][1]
selected_features = [[data_pairs[0][0]['label'], data_pairs[0][2]]]
for pair in data_pairs[1:]:
    pair[1] = pair[1].drop('PLATEIFU', axis=1)
    merge_df = (merge_df.merge(pair[1], left_on='MANGAID', right_on='MANGAID'))
    selected_features += [[pair[0]['label'], pair[2]]]

#print(selected_features)
merge_df.rename(columns={col: col.lower() for col in merge_df.columns}, inplace=True)
print("MERGE LEN: ", len(merge_df))

#Get bpt data and filter out entries with errors
bpt_data = prepare_bpt_data(merge_df)
bpt_data = bpt_data[bpt_data['bpt_error'] == 0]

merge_df = merge_df.drop_duplicates(subset=['plateifu'])
bpt_data = bpt_data.drop_duplicates(subset=['plateifu'])

merge_df = (bpt_data.merge(merge_df, on='plateifu', how='inner' ))

selected_features.append(['BPT', ['bpt_0.0', 'bpt_1.0', 'bpt_2.0', 'bpt_3.0']])

#Remove non-numeric data
numeric_df = get_numeric_df(merge_df)

#Select features of interest
firefly_str = ["lw_age_1re", "mw_age_1re", "lw_z_1re", "mw_z_1re", "redshift", "photometric_mass", "bpt_0.0", "bpt_1.0", "bpt_2.0", "bpt_3.0"]
meta_str = ["ttype", "unsure"]
debiased_columns = [
    col for col in numeric_df.columns if "debiased" in col #Galaxy Zoo
    or [s for s in firefly_str if s.lower() in col.lower() and "error" not in col.lower()] #Firefly
    or [s for s in meta_str if s.lower() in col.lower() and "error" not in col.lower()] #Meta data
    ]

numeric_df = numeric_df[debiased_columns]

#Scale data before PCA
scaler = MaxAbsScaler()
scaled_df = scaler.fit_transform(numeric_df)

#Merge PCA back into original data
merge_df = merge_df.reset_index(drop=True)
numeric_df = numeric_df.reset_index(drop=True)

dim_red_df,PLOT_XY = run_pca(scaled_df)
CURRENT_EMBEDDING = 'pca' 

merge_df = pd.concat([numeric_df, merge_df['mangaid'], dim_red_df], axis=1, ignore_index=False)
#merge_df = pd.concat([numeric_df, merge_df['mangaid'], merge_df['plateifu'], dim_red_df], axis=1, ignore_index=False)

merge_df['cluster'] = run_agglomerative(dim_red_df, 3) 
CURRENT_CLUSTERING = 'agglomerative' 

df = merge_df

excluded_labels = ['mangaid', 'plateifu', 'pc1', 'pc2', 'tsne1', 'tsne2', 'iso1', 'iso2']

label_df = df.drop(excluded_labels, axis=1, errors='ignore')

app = dash.Dash(__name__)

embedding_options=['pca', 'tsne']
clustering_options=['agglomerative', 'hdbscan', 'kmeans', 'meanshift']

app.layout = get_page_layout(label_df, embedding_options, clustering_options, firefly_str, selected_features)

# Callback to handle header checkbox changes
@app.callback(
    Output('galaxy-zoo-checklist', 'value'),
    Output('firefly-checklist', 'value'),
    Input('galaxy-zoo-check-header', 'value'),
    Input('firefly-check-header', 'value'),
    prevent_initial_call=True,
)
def update_checklists(galaxy_zoo_header, firefly_header):
    # Check or uncheck all checkboxes based on header checkbox state
    galaxy_zoo = label_df.columns[label_df.columns.str.contains('debiased')].tolist()

    return galaxy_zoo if galaxy_zoo_header else [], firefly_str if firefly_header else []

# Callback to hide/show the embedding parameters based on choice 
@app.callback(
    Output('embedding-param-container', 'style'),
    Input('embedding-selector', 'value'),
)
def update_embedding_param_visibility(selected_option):
    return {'display': 'none'} if selected_option != 'tsne' else {'display': 'block'}

#TODO: agglomerative doesn't take a random seed, should remove it from input options
@app.callback(
    Output('clustering-param-container', 'style'),
    Input('clustering-selector', 'value'),
)
def update_embedding_param_visibility(selected_option):
    return {'display': 'none'} if selected_option != 'kmeans' and selected_option != 'agglomerative' else {'display': 'block'}

@app.callback(
    Output('scatterplot', 'figure'),
    Output('clusterscatter', 'figure'),
    Output('barplot', 'figure'),
    Output('clusterline', 'figure'),
    Output('regen-button', 'n_clicks'),
    Output('regen-cluster-button', 'n_clicks'),

    Input('color-selector', 'value'),
    Input('regen-button', 'n_clicks'),
    Input('regen-cluster-button', 'n_clicks'),

    State('embedding-selector', 'value'),
    State('perplexity-input', 'value'),
    State('tsne-seed-input', 'value'),
    State('clustering-selector', 'value'),
    State('k-input', 'value'),
    State('k-seed-input', 'value'),
    State('features-list', 'children'),
)
def update_scatterplot(selected_color, embedding_n_clicks, cluster_n_clicks,
                        embedding_choice, perplexity, tsne_seed,
                        clustering_choice, num_k, k_seed, list_values):
    global df
    global PLOT_XY 
    global scaled_df 
    global numeric_df 
    global merge_df 
    global dim_red_df
    global CURRENT_EMBEDDING
    global CURRENT_CLUSTERING 

    if ctx.triggered_id == "regen-button" and embedding_n_clicks:
        print("Running Embedding + Clustering:")
        start_time = time.time()
        features = []
        for c in (list_values[2]['props']['children'][1:]):
            features += (c['props']['children'][1]['props']['value'])
        print("Features selected: ", features)

        dim_red_df,PLOT_XY = run_embedding(numeric_df, features, embedding_choice, perplexity, tsne_seed)
        CURRENT_EMBEDDING = embedding_choice

        merge_df = pd.concat([numeric_df, merge_df['mangaid'], dim_red_df], axis=1, ignore_index=False)
        merge_df['cluster'] = run_clustering(dim_red_df, clustering_choice, num_k, k_seed) 
        end_time = time.time()
        elapsed_time = end_time - start_time 
        print("Embedding + Clustering Time taken : {} s".format(elapsed_time))

        CURRENT_CLUSTERING = clustering_choice 
        df = merge_df 

    elif ctx.triggered_id == "regen-cluster-button" and cluster_n_clicks:
        merge_df['cluster'] = run_clustering(dim_red_df, clustering_choice, num_k, k_seed) 
        CURRENT_CLUSTERING = clustering_choice 
        df = merge_df

    df = df.sort_values(by='cluster', ascending=True)

    fig = get_scatter_fig(df, selected_color, PLOT_XY)
    clusterscatterfig = get_cluster_scatter_fig(df, PLOT_XY)
    cluster_line_fig = get_cluster_line_fig(df)
    barfig = get_cluster_bar_fig(df)

    return fig, clusterscatterfig, barfig, cluster_line_fig, 0, 0

@app.callback(
    Output('scatterplot', 'config'),
    Input('scatterplot', 'clickData'),
)
def click_data_point(clickData):
    if clickData is None:
        return {'editable': True}
    else:
        clicked_point_data = clickData['points'][0]
        
        mangaID = clicked_point_data['hovertext'] 
        url = "https://dr17.sdss.org/marvin/galaxy/" + mangaID.strip() + "/"

        # Check if a URL exists for the clicked point
        if url:
            webbrowser.open_new_tab(url)  # Open the URL in a new tab

        return {'editable': True}

@app.callback(
    Output('clusterscatter', 'config'),
    Input('clusterscatter', 'clickData'),
)
def cluster_click_data_point(clickData):
    if clickData is None:
        return {'editable': True}
    else:
        clicked_point_data = clickData['points'][0]
        
        mangaID = clicked_point_data['hovertext'] 
        url = "https://dr17.sdss.org/marvin/galaxy/" + mangaID.strip() + "/"

        # Check if a URL exists for the clicked point
        if url:
            webbrowser.open_new_tab(url)  # Open the URL in a new tab

        return {'editable': True}

# Callback to export cluster data as csv
@app.callback(
    Output('output-message', 'children'),
    Output('export-cluster-button', 'n_clicks'),
    Input('export-cluster-button', 'n_clicks'),
    prevent_initial_call=True,
)
def update_message(n_clicks):
    global merge_df

    if n_clicks is None:
        return '',0  # Initial state, no message
    else:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y-%m-%d-%H:%M:%S")
        outputfile = f'{CURRENT_EMBEDDING}_{CURRENT_CLUSTERING}_{formatted_timestamp}.csv'

        output_dir = './clusters/'

        create_directory(output_dir)

        output_df = merge_df[['mangaid', 'cluster']]
        output_df.to_csv(output_dir + outputfile)
        return f'Exported to {output_dir + outputfile}',0

if __name__ == '__main__':
    app.run_server(debug=True)
