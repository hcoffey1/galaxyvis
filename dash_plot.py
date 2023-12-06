import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import random
#import numpy as np

import webbrowser

from astropy.io import fits
from astropy.table import Table

from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from scipy import stats

import matplotlib.pyplot as plt

PLOT_XY = []

# Define the path to your FITS file
zoo_auto_file_path = "./data/MaNGA_GZD_auto-v1_0_1.fits"
zoo_df17_file_path = "./data/MaNGA_gz-v2_0_1.fits"
zoo_df15_file_path = "./data/MaNGA_gz-v1_0_1.fits"

dap_all_file_path = "./data/dapall-v3_1_1-3.1.0.fits"

#fits_file_path = "./data/MaNGA_gzUKIDSS-v1_0_1.fits"
agn_fits_path = './data/manga_agn-v1_0_1.fits'
swift_fits_path="./data/SwiM_all_v4.fits"
#swift_fits_path="./data/drpall-v3_1_1.fits"

def read_fits(fits_path):
    dat = Table.read(fits_path, format='fits')
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    df['MANGAID'] = df['MANGAID'].str.decode('utf-8')
    df['MANGAID'] = df['MANGAID'].str.strip()
    return df

def clean_df(df):
    #Remove outliers
    #df = df[df['SFR_1RE'] >= -20] #May cause a crash
    #df = df[df['SFR_1RE'] <= 20] #May cause a crash
    #df = df[df['SFR_TOT'] >= -20] #May cause a crash
    #df = df[df['SFR_TOT'] <= 20] #May cause a crash
    df = df[df['DAPQUAL'] == 0] #May cause a crash
    debiased_columns = [col for col in df.columns if "debiased" in col]
    for col in debiased_columns:
        #df[col] = df[col].apply(lambda x: x if 0 <= x <= 1 else None) 
        df = df[df[col] >= 0] 
        df = df[df[col] <= 1] 

    df = df.dropna(axis=1)
    return df

def get_numeric_df(df):
    #Remove non-numeric data
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=non_numeric_columns)
    return df

def run_pca(df):
    col = ['pc1', 'pc2']
    pca = PCA(n_components=2)
    pca_df = pd.DataFrame(pca.fit_transform(scaled_df), columns=col)
    return pca_df.reset_index(drop=True),col

def run_tsne(df, perplexity=50, seed=42):
    print("Running TSNE, perplexity: {}, seed: {}".format(perplexity, seed))
    col = ['tsne1', 'tsne2']
    tsne_model = None 
    if seed < 0:
        tsne_model = TSNE(n_components=2, perplexity=perplexity)
    else:
        tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=seed)

    tsne_df = pd.DataFrame(tsne_model.fit_transform(df), columns=col)
    return tsne_df.reset_index(drop=True),col

def run_isomap(df):
    col = ['iso1', 'iso2']
    tsne_model = Isomap(n_components=2)
    tsne_df = pd.DataFrame(tsne_model.fit_transform(df), columns=col)
    return tsne_df.reset_index(drop=True),col

zoo_auto_df = read_fits(zoo_auto_file_path)
zoo_df15_df = read_fits(zoo_df15_file_path)
zoo_df17_df = read_fits(zoo_df17_file_path)
swift_df = read_fits(swift_fits_path)
agn_df = read_fits(agn_fits_path)
dapall_df = read_fits(dap_all_file_path)

dapall_df = dapall_df[dapall_df['DAPDONE'] == 1]

#dapall_df['HA_GSIGMA_1RE'] = np.log10(dapall_df['HA_GSIGMA_1RE'])
#print(dapall_df['HA_GSIGMA_1RE'])
#exit(1)
#dapall_df['STELLAR_SIGMA_1RE'] = np.log10(dapall_df['STELLAR_SIGMA_1RE'])

prefix_to_remove = 'manga-'
agn_df['MANGAID'] = agn_df['MANGAID'].str.replace(f'^{prefix_to_remove}', '', regex=True)

#print("AUTO-------------------")
#print(zoo_auto_df)
print("DF15-------------------")
print(zoo_df15_df)
print("DF17-------------------")
print(zoo_df17_df)
print("AGN-------------------")
print(agn_df)
print("SWIFT-------------------")
print(swift_df)
print("DAPALL-------------------")
print(dapall_df)

print('CONCAT------')
zoo_concat = zoo_df17_df
#zoo_concat = pd.concat([zoo_df15_df, zoo_df17_df])

x = (zoo_concat['MANGAID'][10])
print(x + "test")
print("1-106630" + "test")
print(zoo_concat[zoo_concat['MANGAID'] == x])

print("DUPS: ", zoo_concat.duplicated(subset='MANGAID').sum())

#merge_df = (zoo_auto_df.merge(zoo_df15_df, left_on='MANGAID', right_on='MANGAID'))

#print(merge_df)

#merge_df = zoo_concat 
#merge_df = (zoo_concat.merge(swift_df, left_on='MANGAID', right_on='MANGAID'))
merge_df = (zoo_concat.merge(dapall_df, left_on='MANGAID', right_on='MANGAID'))

print("--------------------------------\n")
print(zoo_concat['MANGAID'])
print(dapall_df['MANGAID'])

print("MERGE-------------------")
#Remove outliers
merge_df = clean_df(merge_df)
print(merge_df)

#Remove non-numeric data
numeric_df = get_numeric_df(merge_df)

#print(merge_df['SFR_1RE'])

for col in numeric_df.columns:
    print(col)


#Select debiased columns
debiased_columns = [col for col in numeric_df.columns if "debiased" in col \
       or "1RE" in col]
        #+ ['OBJDEC_x', 'OBJRA_x'] 
        #+ ['SFR_1RE', 'HA_GSIGMA_1RE', 'STELLAR_SIGMA_1RE']
#for c in (debiased_columns):
#    print(c)
#exit(1)
numeric_df = numeric_df[debiased_columns]

#numeric_df = numeric_df[numeric_df['OBJDEC_x'] > 45]

#bool_map = ((np.abs(stats.zscore(numeric_df)) < 3).all(axis=1))
#neg_bool_map = [not elem for elem in bool_map]
#numeric_df = numeric_df[bool_map]

#numeric_df = numeric_df[(np.abs(stats.zscore(numeric_df)) < 3).all(axis=1)]

#numeric_df = numeric_df[debiased_columns + ['SFR_1RE']]

print("Numeric data----------")
#numeric_df = numeric_df.sort_values(by=['SFR_1RE'], ascending=False)
print(numeric_df)

#numeric_df = numeric_df.reset_index()
##print(numeric_df)
#numeric_df.plot.scatter(x='index', y='SFR_1RE')
##plt.plot(y=numeric_df['SFR_1RE'])
#plt.show()
#
#numeric_df.plot.scatter(x='index', y='SFR_TOT')
##plt.plot(y=numeric_df['SFR_1RE'])
#plt.show()

#Scale data before PCA
scaler = MaxAbsScaler()
#scaler = StandardScaler()
scaled_df = scaler.fit_transform(numeric_df)

#Merge PCA back into original data
merge_df = merge_df.reset_index(drop=True)
numeric_df = numeric_df.reset_index(drop=True)

dim_red_df,PLOT_XY = run_pca(scaled_df)
#dim_red_df,PLOT_XY = run_tsne(scaled_df, perplexity=100, seed=100)
#dim_red_df,PLOT_XY = run_isomap(scaled_df)

merge_df = pd.concat([numeric_df, merge_df['MANGAID'], dim_red_df], axis=1, ignore_index=False)

#df = pd.DataFrame(data)
df = merge_df 

app = dash.Dash(__name__)

embedding_options=['pca', 'tsne']

app.layout = html.Div([
    html.H1("MaNGA Galaxy Visualizer"),
    dcc.Dropdown(
        id="color-selector",
        options=[{'label': col, 'value': col} for col in df.columns[2:]],
        value=df.columns[2],  # Initial value
    ),

    html.Div([
        html.Div([
            dcc.Loading(
            id='loading-indicator',
            type='circle',  # or 'default'
            children=[
                dcc.Graph(id='scatterplot'),
                ]
            ),
        ], style={'flex': '1', 'width': '50%'}),

        html.Div([
            html.Button('Regenerate Graph', id='regen-button', n_clicks=0),

            html.Div([
                html.Label("Embedding Method"),
                dcc.Dropdown(id="embedding-selector", 
                            options=[{'label': option, 'value': option} for option in embedding_options],
                            value=embedding_options[0]
                            ),
                html.Div(id='embedding-param-container', children=[
                            html.Div([
                                html.Label("Perplexity:"),
                                dcc.Input(id='perplexity-input', type='number', value=100),
                            ]),
                            html.Div([
                                html.Label("Random Seed (-1 : Random):"),
                                dcc.Input(id='seed-input', type='number', value=-1),
                            ]),
                        ], style={'display': 'none'})
            ], style={'width': '25%'}),

            html.Details([
                html.Summary('Show/Hide Boxes'),
                html.Div([
                    html.H3("Galaxy Zoo"),
                    dcc.Checklist(
                        id='galaxy-zoo-checklist',
                        options=[
                            {'label': col, 'value': col}
                            for col in df.columns if 'debiased' in col
                        ],
                        value=df.columns[df.columns.str.contains('debiased')].tolist(),
                    )
                ]
                ),

                html.Div([
                    html.H3("Other"),
                    dcc.Checklist(
                        id='other-checklist',
                        options=[
                            {'label': col, 'value': col}
                            for col in df.columns if not 'debiased' in col
                        ],
                    )
                ]
                )
            ], style={'width': 'max-content'}),
        ], style={'width': '50%'}),
    ], style={'display': 'flex'}),
])

# Callback to hide/show the embedding parameters based on choice 
@app.callback(
    Output('embedding-param-container', 'style'),
    Input('embedding-selector', 'value'),
)
def update_embedding_param_visibility(selected_option):
    return {'display': 'none'} if selected_option != 'tsne' else {'display': 'block'}

@app.callback(
    Output('scatterplot', 'figure'),
    Output('regen-button', 'n_clicks'),
    Input('color-selector', 'value'),
    Input('regen-button', 'n_clicks'),
    State('embedding-selector', 'value'),
    State('perplexity-input', 'value'),
    State('seed-input', 'value'),
)
def update_scatterplot(selected_color, n_clicks, embedding_choice, perplexity, seed):
    global df
    global PLOT_XY 
    global scaled_df 
    global numeric_df 
    global merge_df 

    if n_clicks:
        if embedding_choice == "pca":
            dim_red_df,PLOT_XY = run_pca(scaled_df)
            merge_df = pd.concat([numeric_df, merge_df['MANGAID'], dim_red_df], axis=1, ignore_index=False)
            df = merge_df 

        elif embedding_choice == "tsne":
            dim_red_df,PLOT_XY = run_tsne(scaled_df, perplexity=perplexity, seed=seed)
            merge_df = pd.concat([numeric_df, merge_df['MANGAID'], dim_red_df], axis=1, ignore_index=False)
            df = merge_df 

    fig = px.scatter(
        df, x=PLOT_XY[0], y=PLOT_XY[1], color=selected_color, color_continuous_scale='Viridis',
        labels={selected_color: selected_color},
        hover_name="MANGAID",
        title='Interactive Scatterplot with Color Selector',
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title=selected_color),
        height=800,
        width=800,
    )

    return fig, 0

@app.callback(
    Output('scatterplot', 'config'),
    Input('scatterplot', 'clickData'),
)
def click_data_point(clickData):
    if clickData is None:
        return {'editable': True}
    else:
        clicked_point_data = clickData['points'][0]
        
        mangaID = df.loc[clicked_point_data['pointIndex'], 'MANGAID']
        url = "https://dr15.sdss.org/marvin/galaxy/" + mangaID.strip() + "/"

        # Check if a URL exists for the clicked point
        if url:
            webbrowser.open_new_tab(url)  # Open the URL in a new tab

        return {'editable': True}


if __name__ == '__main__':
    app.run_server(debug=True)
