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

from src.layout import get_page_layout, get_scatter_fig, get_cluster_scatter_fig, \
    get_cluster_line_fig, get_cluster_bar_fig, swap_layout

from src.data_processor import run_embedding, run_clustering, prepare_data 

PLOT_XY = []

CURRENT_EMBEDDING = None
CURRENT_CLUSTERING = None

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

manga_data, decals_data = prepare_data("./data")
numeric_df = manga_data[0]
merge_df = manga_data[1]
selected_features = manga_data[2]

excluded_labels = ['mangaid', 'pc1', 'pc2', 'tsne1', 'tsne2', 'iso1', 'iso2']

label_df = merge_df.drop(excluded_labels, axis=1, errors='ignore')

embedding_options=['pca', 'tsne']
clustering_options=['agglomerative', 'hdbscan', 'kmeans', 'meanshift']

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = get_page_layout(label_df, embedding_options, clustering_options, selected_features)

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
    prevent_initial_call=True,
)
def update_embedding_param_visibility(selected_option):
    return {'display': 'none'} if selected_option != 'tsne' else {'display': 'block'}

#TODO: agglomerative doesn't take a random seed, should remove it from input options
@app.callback(
    Output('clustering-param-container', 'style'),
    Input('clustering-selector', 'value'),
    prevent_initial_call=True,
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
    prevent_initial_call=True,
)
def update_scatterplot(selected_color, embedding_n_clicks, cluster_n_clicks,
                        embedding_choice, perplexity, tsne_seed,
                        clustering_choice, num_k, k_seed, list_values):
    #global df
    global PLOT_XY 
    global numeric_df 
    global merge_df 
    global CURRENT_EMBEDDING
    global CURRENT_CLUSTERING 

    #Run embedding and clustering
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

    #Run only clustering
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
    prevent_initial_call=True,
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
    prevent_initial_call=True,
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

# Callback to switch layouts based on button click
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')],
)
def display_page(pathname):
    return swap_layout(pathname) 

@app.callback(
    Output('url', 'pathname'),
    [Input('switch-button', 'n_clicks')],
    [State('url', 'pathname')],
)
def switch_layout(n_clicks, current_path):
    if n_clicks is None:
        # No button click yet
        return current_path

    print("Current path:", current_path) 
    if current_path == '/manga' or current_path == '/':
        return '/decals'
    else:
        return '/manga'

if __name__ == '__main__':
    app.run_server(debug=True)
