import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

import webbrowser

from astropy.io import fits
from astropy.table import Table

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define the path to your FITS file
zoo_auto_file_path = "./data/MaNGA_GZD_auto-v1_0_1.fits"
zoo_df17_file_path = "./data/MaNGA_gz-v2_0_1.fits"
zoo_df15_file_path = './data/MaNGA_gz-v1_0_1.fits'
#fits_file_path = "./data/MaNGA_gzUKIDSS-v1_0_1.fits"
agn_fits_path = './data/manga_agn-v1_0_1.fits'
swift_fits_path="./data/SwiM_all_v4.fits"
#swift_fits_path="./data/drpall-v3_1_1.fits"

def read_fits(fits_path):
    dat = Table.read(fits_path, format='fits')
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    df['MANGAID'] = df['MANGAID'].str.decode('utf-8')
    return df

def clean_df(df):
    #Remove outliers
    #df = df[df['SFR_1RE'] >= -20] #May cause a crash
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

zoo_auto_df = read_fits(zoo_auto_file_path)
zoo_df15_df = read_fits(zoo_df15_file_path)
zoo_df17_df = read_fits(zoo_df17_file_path)
swift_df = read_fits(swift_fits_path)
agn_df = read_fits(agn_fits_path)

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

zoo_concat = pd.concat([zoo_df15_df, zoo_df17_df])

print("DUPS: ", zoo_concat.duplicated(subset='MANGAID').sum())

#merge_df = (zoo_auto_df.merge(zoo_df15_df, left_on='MANGAID', right_on='MANGAID'))

#print(merge_df)

merge_df = zoo_concat 
#merge_df = (zoo_concat.merge(swift_df, left_on='MANGAID', right_on='MANGAID'))

print("MERGE-------------------")
#Remove outliers
merge_df = clean_df(merge_df)
print(merge_df)

#Remove non-numeric data
numeric_df = get_numeric_df(merge_df)


#Select debiased columns
debiased_columns = [col for col in numeric_df.columns if "debiased" in col]
#for c in (debiased_columns):
#    print(c)
numeric_df = numeric_df[debiased_columns]
#numeric_df = numeric_df[debiased_columns + ['SFR_1RE']]

print("Numeric data----------")
print(numeric_df)

#Scale data before PCA
scaler = StandardScaler()
scaled_df = scaler.fit_transform(numeric_df)

#PCA
pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit_transform(scaled_df), columns=['pc1', 'pc2'])
tsne_model = TSNE(n_components=2, perplexity=50, random_state=42)
tsne_df = pd.DataFrame(tsne_model.fit_transform(scaled_df), columns=['tsne1', 'tsne2'])

print(merge_df)
print(pca_df)

#Merge PCA back into original data
merge_df = merge_df.reset_index(drop=True)
pca_df = pca_df.reset_index(drop=True)
tsne_df = tsne_df.reset_index(drop=True)
numeric_df = numeric_df.reset_index(drop=True)

#merge_df = pd.concat([merge_df, pca_df, tsne_df], axis=1, ignore_index=False)
merge_df = pd.concat([numeric_df, merge_df['MANGAID'], tsne_df], axis=1, ignore_index=False)

# Sample data (replace this with your data)
data = {
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 3, 4, 5, 6],
    'ColorValue1': [10, 20, 30, 40, 50],
    'ColorValue2': [5, 15, 25, 35, 45]
}

#df = pd.DataFrame(data)
df = merge_df 

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("MaNGA Galaxy Visualizer"),
    dcc.Dropdown(
        id="color-selector",
        options=[{'label': col, 'value': col} for col in df.columns[2:]],
        value=df.columns[2],  # Initial value
    ),
    dcc.Graph(id='scatterplot'),
])

@app.callback(
    Output('scatterplot', 'figure'),
    Input('color-selector', 'value')
)
def update_scatterplot(selected_color):
    fig = px.scatter(
        df, x='tsne1', y='tsne2', color=selected_color, color_continuous_scale='Viridis',
        labels={selected_color: selected_color},
        hover_name="MANGAID",
        title='Interactive Scatterplot with Color Selector',
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title=selected_color),
        height=800,
        width=800,
    )

    return fig

@app.callback(
    Output('scatterplot', 'config'),
    Input('scatterplot', 'clickData'),
)
def click_data_point(clickData):
    print("Click!")
    if clickData is None:
        return {'editable': True}
    else:
        clicked_point_data = clickData['points'][0]
        
        # Assuming 'URL' is a column in your DataFrame containing the URLs
        mangaID = df.loc[clicked_point_data['pointIndex'], 'MANGAID']
        url = "https://dr15.sdss.org/marvin/galaxy/" + mangaID.strip() + "/"

        # Check if a URL exists for the clicked point
        if url:
            webbrowser.open_new_tab(url)  # Open the URL in a new tab

        return {'editable': True}


if __name__ == '__main__':
    app.run_server(debug=True)
