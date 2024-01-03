#Galaxy Vis : Interactive Visualization Tool for MaNGA galaxies
#Hayden Coffey

import dash
from dash.dependencies import Input, Output, State, ALL
from dash import ctx
from io import StringIO
import pandas as pd
import time

from datetime import datetime
import os

import webbrowser

from src.layout import get_page_layout, get_scatter_fig, get_cluster_scatter_fig, \
    get_cluster_line_fig, get_cluster_bar_fig, swap_layout

from src.data_processor import run_embedding, run_clustering, prepare_data 

from src.decals_callback import decals_callbacks

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

manga_data, decals_data = prepare_data("./data")
manga_numeric_df = manga_data[0]
manga_merge_df = manga_data[1]
manga_selected_features = manga_data[2]

decals_numeric_df = decals_data[0]
decals_merge_df = decals_data[1]
decals_selected_features = decals_data[2]

excluded_labels = ['mangaid', 'pc1', 'pc2', 'tsne1', 'tsne2', 'iso1', 'iso2']

label_df = manga_merge_df.drop(excluded_labels, axis=1, errors='ignore')
label_df_decals = decals_merge_df.drop(excluded_labels, axis=1, errors='ignore')

embedding_options=['pca', 'tsne']
clustering_options=['agglomerative', 'hdbscan', 'kmeans', 'meanshift']

#    return [decals_merge_df.reset_index().to_json(orient='split')]
app = dash.Dash(__name__, suppress_callback_exceptions=False)
app.layout = get_page_layout(label_df, label_df_decals, embedding_options, clustering_options, manga_selected_features, decals_selected_features, decals_merge_df)
decals_callbacks(app)

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
    Output('manga-embedding-param-container', 'style'),
    Input('manga-embedding-selector', 'value'),
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
    Output('manga-scatterplot', 'figure'),
    Output('clusterscatter', 'figure'),
    Output('barplot', 'figure'),
    Output('clusterline', 'figure'),
    Output('manga-regen-button', 'n_clicks'),
    Output('regen-cluster-button', 'n_clicks'),

    Input('manga-color-selector', 'value'),
    Input('manga-regen-button', 'n_clicks'),
    Input('regen-cluster-button', 'n_clicks'),

    State('manga-embedding-selector', 'value'),
    State('manga-perplexity-input', 'value'),
    State('manga-tsne-seed-input', 'value'),
    State('clustering-selector', 'value'),
    State('k-input', 'value'),
    State('k-seed-input', 'value'),
    State('manga-features-list', 'children'),
    prevent_initial_call=True,
)
def update_scatterplot(selected_color, embedding_n_clicks, cluster_n_clicks,
                        embedding_choice, perplexity, tsne_seed,
                        clustering_choice, num_k, k_seed, list_values):
    #global df
    global PLOT_XY 
    global manga_numeric_df 
    global manga_merge_df 
    global CURRENT_EMBEDDING
    global CURRENT_CLUSTERING 

    #Run embedding and clustering
    if ctx.triggered_id == "manga-regen-button" and embedding_n_clicks:
        print("Running Embedding + Clustering:")
        start_time = time.time()
        features = []
        for c in (list_values[2]['props']['children'][1:]):
            features += (c['props']['children'][1]['props']['value'])
        print("Features selected: ", features)

        dim_red_df,PLOT_XY = run_embedding(manga_numeric_df, features, embedding_choice, perplexity, tsne_seed)
        CURRENT_EMBEDDING = embedding_choice

        manga_merge_df = pd.concat([manga_numeric_df, manga_merge_df['mangaid'], dim_red_df], axis=1, ignore_index=False)
        manga_merge_df['cluster'] = run_clustering(dim_red_df, clustering_choice, num_k, k_seed) 
        end_time = time.time()
        elapsed_time = end_time - start_time 
        print("Embedding + Clustering Time taken : {} s".format(elapsed_time))

        CURRENT_CLUSTERING = clustering_choice 

    #Run only clustering
    elif ctx.triggered_id == "regen-cluster-button" and cluster_n_clicks:
        manga_merge_df['cluster'] = run_clustering(dim_red_df, clustering_choice, num_k, k_seed) 
        CURRENT_CLUSTERING = clustering_choice 

    df = manga_merge_df
    df = df.sort_values(by='cluster', ascending=True)

    fig = get_scatter_fig(df, selected_color, PLOT_XY, 'mangaid')
    clusterscatterfig = get_cluster_scatter_fig(df, PLOT_XY)
    cluster_line_fig = get_cluster_line_fig(df)
    barfig = get_cluster_bar_fig(df)

    return fig, clusterscatterfig, barfig, cluster_line_fig, 0, 0


@app.callback(
    Output('manga-scatterplot', 'config'),
    Input('manga-scatterplot', 'clickData'),
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
    Input('current-embedding', 'data'),
    Input('current-clustering', 'data'),
    prevent_initial_call=True,
)
def update_message(n_clicks, cur_embedding, cur_clustering):
    global manga_merge_df


    if n_clicks is None:
        return '',0  # Initial state, no message
    else:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y-%m-%d-%H:%M:%S")
        outputfile = f'{cur_embedding}_{cur_clustering}_{formatted_timestamp}.csv'

        output_dir = './clusters/'

        create_directory(output_dir)

        output_df = manga_merge_df[['mangaid', 'cluster']]
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
