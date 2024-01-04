#Galaxy Vis : Interactive Visualization Tool for MaNGA galaxies
#Hayden Coffey

import dash
from dash.dependencies import Input, Output, State 

from src.layout import get_page_layout, swap_layout

from src.data_processor import prepare_data 

from src.decals_callback import decals_callbacks
from src.manga_callback import manga_callbacks

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

app = dash.Dash(__name__, suppress_callback_exceptions=False)
app.layout = get_page_layout(label_df, label_df_decals, embedding_options, clustering_options, manga_selected_features, decals_selected_features, decals_merge_df, manga_merge_df)
decals_callbacks(app)
manga_callbacks(app)

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
