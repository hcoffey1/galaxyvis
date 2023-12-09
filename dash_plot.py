#MaNGA Galaxy Visualizer
#Hayden Coffey

import dash
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import time

import webbrowser

from astropy.table import Table

from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler

from sklearn.cluster import KMeans

from layout import get_page_layout

PLOT_XY = []

# Define the path to your FITS file
zoo_df17_file_path = "./data/MaNGA_gz-v2_0_1.fits"
firefly_file_path= "./data/manga-firefly-globalprop-v3_1_1-mastar.fits"
dap_all_file_path = "./data/dapall-v3_1_1-3.1.0.fits"

def read_fits(fits_path, hdu=1):
    dat = Table.read(fits_path, format='fits', hdu=hdu)
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    if 'MANGAID' in df:
        df['MANGAID'] = df['MANGAID'].str.decode('utf-8')
        df['MANGAID'] = df['MANGAID'].str.strip()
    return df

def clean_df(df):
    #Remove outliers
    if 'SFR_1RE' in df:
        df = df[df['SFR_1RE'] >= -20]
        df = df[df['SFR_1RE'] <= 20] 
    if 'SFR_TOT' in df:
        df = df[df['SFR_TOT'] >= -20]
        df = df[df['SFR_TOT'] <= 20] 
    if 'DAPQUAL' in df:
        df = df[df['DAPQUAL'] == 0] 

    debiased_columns = [col for col in df.columns if "debiased" in col]
    for col in debiased_columns:
        df = df[df[col] >= 0] 
        df = df[df[col] <= 1] 

    df = df.dropna(axis=1)

    df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)

    rows_to_remove = df.map(lambda x: isinstance(x, (int, float)) and x < -9000).any(axis=1)
    df = df[~rows_to_remove] #filter out errors from firefly

    return df

def get_numeric_df(df):
    #Remove non-numeric data
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=non_numeric_columns)
    return df

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

def run_kmeans(df, k, seed=42):
    if seed < 0:
        kmeans = KMeans(n_clusters=k, n_init=10)
    else:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)

    kmeans.fit(df)
    return kmeans.labels_

#Read in fits files
zoo_df17_df = read_fits(zoo_df17_file_path)

dapall_df = read_fits(dap_all_file_path)
dapall_df = dapall_df[dapall_df['DAPDONE'] == 1]

firefly_hdu_1_df = read_fits(firefly_file_path)
firefly_hdu_2_df = read_fits(firefly_file_path,2)
firefly_df = pd.concat([firefly_hdu_1_df, firefly_hdu_2_df], axis=1)

#Merge dataframes
merge_df = (zoo_df17_df.merge(firefly_df, left_on='MANGAID', right_on='MANGAID'))
merge_df = (merge_df.merge(dapall_df, left_on='MANGAID', right_on='MANGAID'))

#Remove outliers
merge_df = clean_df(merge_df)

#Remove non-numeric data
numeric_df = get_numeric_df(merge_df)

#Select features of interest
firefly_str = ["lw_age_1re", "mw_age_1re", "lw_z_1re", "mw_z_1re", "redshift", "photometric_mass"]
debiased_columns = [
    col for col in numeric_df.columns if "debiased" in col #Galaxy Zoo
    or [s for s in firefly_str if s.lower() in col.lower() and "error" not in col.lower()] #Firefly
    ]

numeric_df = numeric_df[debiased_columns]

#Scale data before PCA
scaler = MaxAbsScaler()
scaled_df = scaler.fit_transform(numeric_df)

#Merge PCA back into original data
merge_df = merge_df.reset_index(drop=True)
numeric_df = numeric_df.reset_index(drop=True)

dim_red_df,PLOT_XY = run_pca(scaled_df)

merge_df = pd.concat([numeric_df, merge_df['mangaid'], dim_red_df], axis=1, ignore_index=False)

merge_df['cluster'] = run_kmeans(dim_red_df, 3) 

df = merge_df

excluded_labels = ['mangaid', 'pc1', 'pc2', 'tsne1', 'tsne2', 'iso1', 'iso2']

label_df = df.drop(excluded_labels, axis=1, errors='ignore')

app = dash.Dash(__name__)

embedding_options=['pca', 'tsne']
clustering_options=['kmeans']

app.layout = get_page_layout(label_df, embedding_options, clustering_options, firefly_str)

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

@app.callback(
    Output('clustering-param-container', 'style'),
    Input('clustering-selector', 'value'),
)
def update_embedding_param_visibility(selected_option):
    return {'display': 'none'} if selected_option != 'kmeans' else {'display': 'block'}

@app.callback(
    Output('scatterplot', 'figure'),
    Output('clusterscatter', 'figure'),
    Output('barplot', 'figure'),
    Output('regen-button', 'n_clicks'),
    Input('color-selector', 'value'),
    Input('regen-button', 'n_clicks'),
    State('embedding-selector', 'value'),
    State('perplexity-input', 'value'),
    State('tsne-seed-input', 'value'),
    State('galaxy-zoo-checklist', 'value'),
    State('firefly-checklist', 'value'),
    State('k-input', 'value'),
    State('k-seed-input', 'value'),
)
def update_scatterplot(selected_color, n_clicks, embedding_choice, perplexity, tsne_seed, galaxy_zoo_list, firefly_list, num_k, k_seed):
    global df
    global PLOT_XY 
    global scaled_df 
    global numeric_df 
    global merge_df 

    if n_clicks:
        features = galaxy_zoo_list + firefly_list

        #TODO: If embedding stays the same and only clustering changes, don't run embedding again
        if embedding_choice == "pca":
            scaled_df = scaler.fit_transform(numeric_df[features])
            dim_red_df,PLOT_XY = run_pca(scaled_df)
            merge_df = pd.concat([numeric_df, merge_df['mangaid'], dim_red_df], axis=1, ignore_index=False)
            merge_df['cluster'] = run_kmeans(dim_red_df, num_k, k_seed) 
            df = merge_df 

        elif embedding_choice == "tsne":
            scaled_df = scaler.fit_transform(numeric_df[features])
            dim_red_df,PLOT_XY = run_tsne(scaled_df, perplexity=perplexity, seed=tsne_seed)
            merge_df = pd.concat([numeric_df, merge_df['mangaid'], dim_red_df], axis=1, ignore_index=False)
            merge_df['cluster'] = run_kmeans(dim_red_df, num_k, k_seed) 
            df = merge_df 

    fig = px.scatter(
        df, x=PLOT_XY[0], y=PLOT_XY[1], color=selected_color, color_continuous_scale='Viridis',
        labels={selected_color: selected_color},
        hover_name="mangaid",
        title='Interactive Scatterplot with Color Selector',
    )

    if selected_color == 'cluster':
        fig.update_coloraxes(showscale=False)

    fig.update_layout(
        coloraxis_colorbar=dict(title=selected_color),
        height=800,
    )

    clusterscatterfig = px.scatter(
        df, x=PLOT_XY[0], y=PLOT_XY[1], color='cluster', color_continuous_scale='Viridis',
        labels={'cluster': 'cluster'},
        hover_name="mangaid",
        title='Clusters in Embedding Space',
    )
    clusterscatterfig.update_coloraxes(showscale=False)
    clusterscatterfig.update_layout(
        height=400,
        width=400,
        margin=dict(l=0)
    )
    clusterscatterfig.update_xaxes(showticklabels=False, title_text='')
    clusterscatterfig.update_yaxes(showticklabels=False, title_text='')

    cluster_perc = pd.DataFrame(df['cluster'].value_counts(normalize=True)*100).reset_index()
    cluster_perc['group'] = ''
    cluster_perc['cluster'] = cluster_perc['cluster']

    barfig =px.bar(
            cluster_perc,
            y='group',
            barmode='stack',
            x='proportion',
            color='cluster',
            orientation='h',
            custom_data=['proportion', 'cluster'],
            color_continuous_scale='Viridis',
        ).update_xaxes(range=[0, 100])  # Set y-axis range from 0 to 100%

    #TODO: Move clustering method down to here and create separate layout section for cluster analysis?
    #Stacked proportion bar chart
    barfig.update_traces(
        hovertemplate="<b>cluster:</b> %{customdata[1]}<extra></extra><br><b>%{customdata[0]:.2f}%</b>"
    )
    barfig.update_layout(
        coloraxis_colorbar=dict(title=selected_color),
        height=200,
        title_text='Cluster %',
    )
    barfig.update_coloraxes(showscale=False)

    barfig.update_xaxes(title_text='Proportion')
    barfig.update_yaxes(title_text='')


    return fig, clusterscatterfig, barfig, 0

@app.callback(
    Output('scatterplot', 'config'),
    Input('scatterplot', 'clickData'),
)
def click_data_point(clickData):
    if clickData is None:
        return {'editable': True}
    else:
        clicked_point_data = clickData['points'][0]
        
        mangaID = df.loc[clicked_point_data['pointIndex'], 'mangaid']
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
        
        mangaID = df.loc[clicked_point_data['pointIndex'], 'mangaid']
        url = "https://dr17.sdss.org/marvin/galaxy/" + mangaID.strip() + "/"

        # Check if a URL exists for the clicked point
        if url:
            webbrowser.open_new_tab(url)  # Open the URL in a new tab

        return {'editable': True}


if __name__ == '__main__':
    app.run_server(debug=True)
