from dash.dependencies import Input, Output, State
from dash import ctx
from io import StringIO
import pandas as pd
import time

from datetime import datetime
import os

import webbrowser

from src.layout import get_scatter_fig, get_cluster_scatter_fig, \
    get_cluster_line_fig, get_cluster_bar_fig 

from src.data_processor import run_embedding, run_clustering 

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def manga_callbacks(app):
	@app.callback(
		Output('manga-scatterplot', 'figure'),
		Output('clusterscatter', 'figure'),
		Output('barplot', 'figure'),
		Output('clusterline', 'figure'),
		Output('manga-regen-button', 'n_clicks'),
		Output('regen-cluster-button', 'n_clicks'),

		Output('manga-embedding-data', 'data'),
		Output('manga-current-embedding', 'data'),
		Output('manga-current-clustering', 'data'),
		Output('manga-current-xy', 'data'),

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

		State('manga-data', 'data'),
		State('manga-embedding-data', 'data'),
		State('manga-current-embedding', 'data'),
		State('manga-current-clustering', 'data'),
		State('manga-current-xy', 'data'),
		prevent_initial_call=True,
	)
	def update_scatterplot(selected_color, embedding_n_clicks, cluster_n_clicks,
							embedding_choice, perplexity, tsne_seed,
							clustering_choice, num_k, k_seed, list_values, manga_data, embedding_data, cur_embedding, cur_clustering, cur_xy):

		manga_merge_df = (pd.read_json(StringIO(manga_data), orient='split'))
		manga_numeric_df = manga_merge_df.drop("mangaid", axis=1)

		dim_red_df = pd.DataFrame()
		updated_embeddings = embedding_data 
		fig = None
		#global CURRENT_CLUSTERING 

		if embedding_data:
			dim_red_df = (pd.read_json(StringIO(embedding_data), orient='split'))

		#Run embedding and clustering
		if ctx.triggered_id == "manga-regen-button" and embedding_n_clicks:
			print("Running Embedding + Clustering:")
			start_time = time.time()
			features = []
			for c in (list_values[2]['props']['children'][1:]):
				features += (c['props']['children'][1]['props']['value'])
			print("Features selected: ", features)

			dim_red_df,cur_xy= run_embedding(manga_numeric_df, features, embedding_choice, perplexity, tsne_seed)
			updated_embeddings = dim_red_df.reset_index(drop=True).to_json(orient='split')
			cur_embedding = embedding_choice

			manga_merge_df = pd.concat([manga_numeric_df, manga_merge_df['mangaid'], dim_red_df], axis=1, ignore_index=False)
			manga_merge_df['cluster'] = run_clustering(dim_red_df, clustering_choice, num_k, k_seed) 
			end_time = time.time()
			elapsed_time = end_time - start_time 
			print("Embedding + Clustering Time taken : {} s".format(elapsed_time))

			cur_clustering = clustering_choice 

		#Run only clustering
		elif ctx.triggered_id == "regen-cluster-button" and cluster_n_clicks:
			manga_merge_df['cluster'] = run_clustering(dim_red_df, clustering_choice, num_k, k_seed) 
			cur_clustering = clustering_choice 
		else:
			if not dim_red_df.empty:
				manga_merge_df = pd.concat([manga_merge_df, dim_red_df], axis=1, ignore_index=False)
				updated_embeddings = dim_red_df.reset_index(drop=True).to_json(orient='split')

		df = manga_merge_df
		if 'cluster' in df:
			df = df.sort_values(by='cluster', ascending=True)

		fig = get_scatter_fig(df, selected_color, cur_xy, 'mangaid')
		clusterscatterfig = get_cluster_scatter_fig(df, cur_xy)
		cluster_line_fig = get_cluster_line_fig(df)
		barfig = get_cluster_bar_fig(df)

		return fig, clusterscatterfig, barfig, cluster_line_fig, 0, 0, updated_embeddings, cur_embedding, cur_clustering, cur_xy 

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

	# Callback to export cluster data as csv
	@app.callback(
		Output('output-message', 'children'),
		Output('export-cluster-button', 'n_clicks'),
		Input('export-cluster-button', 'n_clicks'),
		Input('manga-current-embedding', 'data'),
		Input('manga-current-clustering', 'data'),
		State('manga-data', 'data'),
		prevent_initial_call=True,
	)
	def update_message(n_clicks, cur_embedding, cur_clustering, manga_data):
		manga_merge_df = (pd.read_json(StringIO(manga_data), orient='split'))


		if n_clicks is None:
			return '',0  # Initial state, no message
		else:

			if 'cluster' not in manga_merge_df:
				return '',0  # Initial state, no message

			timestamp = datetime.now()
			formatted_timestamp = timestamp.strftime("%Y-%m-%d-%H:%M:%S")
			outputfile = f'{cur_embedding}_{cur_clustering}_{formatted_timestamp}.csv'

			output_dir = './clusters/'

			create_directory(output_dir)

			output_df = manga_merge_df[['mangaid', 'cluster']]
			output_df.to_csv(output_dir + outputfile)
			return f'Exported to {output_dir + outputfile}',0