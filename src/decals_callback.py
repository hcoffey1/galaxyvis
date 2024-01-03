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

def decals_callbacks(app):
	@app.callback(
		Output('decals-scatterplot', 'figure'),
		#Output('clusterscatter', 'figure'),
		#Output('barplot', 'figure'),
		#Output('clusterline', 'figure'),
		Output('decals-regen-button', 'n_clicks'),
		Output('decals-embedding-data', 'data'),
		Output('current-embedding', 'data'),
		Output('current-clustering', 'data'),
		Output('current-xy', 'data'),
		#Output('regen-cluster-button', 'n_clicks'),

		Input('decals-color-selector', 'value'),
		Input('decals-regen-button', 'n_clicks'),
		#Input('regen-cluster-button', 'n_clicks'),

		State('decals-embedding-selector', 'value'),
		State('decals-perplexity-input', 'value'),
		State('decals-tsne-seed-input', 'value'),
		#State('clustering-selector', 'value'),
		#State('k-input', 'value'),
		#State('k-seed-input', 'value'),
		State('decals-features-list', 'children'),
		State('decals-data', 'data'),
		State('decals-embedding-data', 'data'),
		State('current-embedding', 'data'),
		State('current-clustering', 'data'),
		State('current-xy', 'data'),
		#State('decals-plot-xy', 'data'),
		prevent_initial_call=True,
	)
	def update_scatterplot(selected_color, embedding_n_clicks,
							embedding_choice, perplexity, tsne_seed,
							list_values,decals_data, decals_embedding_data, cur_embedding, cur_clustering, cur_xy):
		decals_merge_df = (pd.read_json(StringIO(decals_data), orient='split'))
		decals_numeric_df = decals_merge_df.drop("iauname", axis=1)

		if decals_embedding_data:
			dim_red_df = (pd.read_json(StringIO(decals_embedding_data), orient='split'))

		#Run embedding and clustering
		if ctx.triggered_id == "decals-regen-button" and embedding_n_clicks:
			print("Running Embedding + Clustering:")
			start_time = time.time()
			features = []
			for c in (list_values[2]['props']['children'][1:]):
				features += (c['props']['children'][1]['props']['value'])
			print("Features selected: ", features)

			dim_red_df,cur_xy = run_embedding(decals_numeric_df, features, embedding_choice, perplexity, tsne_seed)
			cur_embedding = embedding_choice

			decals_merge_df = pd.concat([decals_numeric_df, decals_merge_df['iauname'], dim_red_df], axis=1, ignore_index=False)
			#merge_df['cluster'] = run_clustering(dim_red_df, clustering_choice, num_k, k_seed) 
			end_time = time.time()
			elapsed_time = end_time - start_time 
			print("Embedding + Clustering Time taken : {} s".format(elapsed_time))

			#cur_clustering = clustering_choice 
		else:
			decals_merge_df = pd.concat([decals_merge_df, dim_red_df], axis=1, ignore_index=False)

		#Run only clustering
		#elif ctx.triggered_id == "regen-cluster-button" and cluster_n_clicks:
		#    merge_df['cluster'] = run_clustering(dim_red_df, clustering_choice, num_k, k_seed) 
		#    cur_clustering = clustering_choice 

		df = decals_merge_df
		#df = df.sort_values(by='cluster', ascending=True)

		fig = get_scatter_fig(df, selected_color, cur_xy, 'iauname')
		#clusterscatterfig = get_cluster_scatter_fig(df, PLOT_XY)
		#cluster_line_fig = get_cluster_line_fig(df)
		#barfig = get_cluster_bar_fig(df)

		return fig, 0, dim_red_df.reset_index(drop=True).to_json(orient='split'), cur_embedding, cur_clustering, cur_xy
		#return fig, clusterscatterfig, barfig, cluster_line_fig, 0, 0