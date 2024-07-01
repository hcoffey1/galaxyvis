#Web page layout
#Hayden Coffey

from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

TITLE = "Galaxy Vis"

def create_features_list(features):

    divs = []

    for f in features:
        label = f[0].strip("\"")
        id = label.lower().replace(" ","-")

        columns = f[1]
        print(label)
        print(id)
        list_div = html.Div([
                html.Div([
                    html.B(label, style={'flex': '1'}),
                    dcc.Checklist(
                        id= id + '-check-header',
                        options=[
                            {'label' : '', 'value': 'ticked'}
                        ],
                        value=['ticked']
                    ),
                ], style={'display': 'flex', 'width' : 'max-content'}),
                dcc.Checklist(
                    id= id + '-checklist',
                    options=[
                        {'label': col, 'value': col}
                        for col in columns
                    ],
                    value=columns,
                )
            ])
        
        divs.append(list_div)

    return divs 


def get_scatter_fig(df, selected_color, PLOT_XY):
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

    return fig

def get_cluster_scatter_fig(df, PLOT_XY):
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
    
    return clusterscatterfig

def get_cluster_line_fig(df):
    grouped = df.groupby(['cluster', 'ttype']).size().reset_index(name='count')

    # Calculate the total count for each cluster
    cluster_totals = grouped.groupby('cluster')['count'].sum()
    ttype_totals = (pd.DataFrame( grouped.groupby('ttype')['count'].sum()).reset_index())

    # Merge the grouped data with the total counts
    result = pd.merge(grouped, cluster_totals, on='cluster', suffixes=('_class', '_total'))

    # Calculate the percentage for each classification within each cluster
    result['percentage'] = (result['count_class'] / result['count_total']) * 100
    
    # Create traces for each cluster
    traces = []
    for cluster in result['cluster'].unique():
        cluster_data = result[result['cluster'] == cluster]
        trace = go.Scatter(x=cluster_data['ttype'], y=cluster_data['count_class'],
                        mode='lines+markers', name=f'Cluster {cluster}')
        traces.append(trace)

    # Add a trace for the 'Total'
    total_trace = go.Scatter(x=ttype_totals['ttype'], y=ttype_totals['count'],
                            mode='lines+markers', name='Total', line=dict(dash='dash'))
    traces.append(total_trace)

    # Create the layout
    layout = go.Layout(
                    xaxis=dict(title='T-Type'),
                    yaxis=dict(title='Galaxy Count'))

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    return fig

def get_cluster_bar_fig(df):
    cluster_perc = pd.DataFrame(df['cluster'].value_counts(normalize=True)*100).reset_index()
    cluster_perc['group'] = ''
    cluster_perc['cluster'] = cluster_perc['cluster']

    cluster_perc = cluster_perc.sort_values(by='cluster', ascending=True)

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
        #coloraxis_colorbar=dict(title=selected_color),
        height=200,
        title_text='Cluster %',
    )
    barfig.update_coloraxes(showscale=False)

    barfig.update_xaxes(title_text='Proportion')
    barfig.update_yaxes(title_text='')

    return barfig


def get_page_layout(label_df, embedding_options, clustering_options, firefly_str, features):
    highlight_feature_div = html.Div([
            html.B("Highlighted Feature:"),
            dcc.Dropdown(
                id="color-selector",
                options=[{'label': col, 'value': col} for col in label_df.columns],
                value=label_df.columns[0],  # Initial value
            ),
        ], style={'width': '50%'})

    scatter_plot_div = html.Div([
        dcc.Loading(
            id='loading-indicator',
            type='circle',  # or 'default'
            children=[
                dcc.Graph(id='scatterplot'),
                ]
            ),
        ], style={'flex': '1', 'width': '50%'})
    
    embedding_method_div = html.Div([
            html.B("Embedding Method"),
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
                    dcc.Input(id='tsne-seed-input', type='number', value=-1),
                ]),
            ], style={'display': 'none'})
        ], style={'width': '25%'})

    clustering_method_div = html.Div([
            html.B("Clustering Method"),
            dcc.Dropdown(id="clustering-selector", 
            options=[{'label': option, 'value': option} for option in clustering_options],
            value=clustering_options[0]
            ),
            html.Div(id='clustering-param-container', children=[
                html.Div([
                    html.Label("k: "),
                    dcc.Input(id='k-input', type='number', value=3),
                ]),
                html.Div([
                    html.Label("Random Seed (-1 : Random):"),
                    dcc.Input(id='k-seed-input', type='number', value=-1),
                ]),
            ], style={'display': 'none'})
        ], style={'width': '25%'})
    
    clustering_div = html.Div([
        clustering_method_div,
        dcc.Loading(
            id='loading-indicator-embeddings',
            type='circle',  # or 'default'
            children=[
                html.Div([
                dcc.Graph(id='clusterscatter', style={'flex': '1'}),
                dcc.Graph(id='clusterline'),
                ], style={'display': 'flex'}),
                dcc.Graph(id='barplot'),
                ]
            ),
    ], style={'width': '50%'})

    features_list = create_features_list(features)

    features_list.insert(0, html.Summary('Input Features'))

    galaxy_zoo_list_div = html.Div([
            html.Div([
                html.B("Galaxy Zoo", style={'flex': '1'}),
                dcc.Checklist(
                    id='galaxy-zoo-check-header',
                    options=[
                        {'label' : '', 'value': 'ticked'}
                    ],
                    value=['ticked']
                ),
            ], style={'display': 'flex', 'width' : 'max-content'}),
            dcc.Checklist(
                id='galaxy-zoo-checklist',
                options=[
                    {'label': col, 'value': col}
                    for col in label_df.columns if 'debiased' in col
                ],
                value=label_df.columns[label_df.columns.str.contains('debiased')].tolist(),
            )
        ])

    firefly_list_div = html.Div([
            html.Div([
                html.B("Firefly", style={'flex': '1'}),
                dcc.Checklist(
                    id='firefly-check-header',
                    options=[
                        {'label' : '', 'value': 'ticked'}
                    ],
                    value=['ticked']
                ),
            ], style={'display': 'flex', 'width' :'max-content'}),
            dcc.Checklist(
                id='firefly-checklist',
                options=[
                    {'label': col, 'value': col}
                    for col in firefly_str 
                ],
                value=firefly_str,
            )
        ])

    return html.Div([
        html.H1(TITLE),
        html.P("Galaxy Count: {}".format(label_df.shape[0])),

        highlight_feature_div,

        html.Div([
            scatter_plot_div,

            html.Div([
                html.Button('Regenerate Graph', id='regen-button', n_clicks=0),
                embedding_method_div,

                html.Details(
                   features_list 
                , style={'width': 'max-content'}),
            ], style={'width': '50%'}, id='features-list'),
        ], style={'display': 'flex'}),
        html.Button('Calculate Clusters', id='regen-cluster-button', n_clicks=0),
        clustering_div,
        html.Div([
            html.Button('Export Cluster Data', id='export-cluster-button', n_clicks=0, style={'width' : 'max-content'}),
            html.Div(id='output-message', style={'flex' : '1'}),
        ], style={'display':'flex'}),
    ])