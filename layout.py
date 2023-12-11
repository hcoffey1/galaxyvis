#Web page layout
#Hayden Coffey

from dash import dcc, html


def get_page_layout(label_df, embedding_options, clustering_options, firefly_str):
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
		html.H1("MaNGA Galaxy Visualizer"),
		html.P("Galaxy Count: {}".format(label_df.shape[0])),

		highlight_feature_div,

		html.Div([
			scatter_plot_div,

			html.Div([
				html.Button('Regenerate Graph', id='regen-button', n_clicks=0),
				embedding_method_div,

				html.Details([
					html.Summary('Input Features'),
					galaxy_zoo_list_div,
					firefly_list_div,
				], style={'width': 'max-content'}),
			], style={'width': '50%'}),
		], style={'display': 'flex'}),
		html.Button('Calculate Clusters', id='regen-cluster-button', n_clicks=0),
		clustering_div,
	])