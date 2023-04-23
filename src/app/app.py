import streamlit as st
import pickle
from kloppy import secondspectrum
import sys
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
sys.path.append('../..')
import src.visualizations as viz
from src.second_spectrum_utils import get_home_away_tracking
import src.player_velocities as vel


META_PATH = "../../man-city-data/SecondSpectrum/MCI Women_s Files-001/g2312166_SecondSpectrum_meta.xml"
RAW_PATH = "../../man-city-data/SecondSpectrum/MCI Women_s Files-001/g2312166_SecondSpectrum_tracking-produced.jsonl"
EVENT_PATH = "../../man-city-data/StatsBomb/Data/ManCity_Liverpool_events.json"


@st.cache_data
def load_data():
    dataset = secondspectrum.load(
        meta_data=META_PATH,
        raw_data=RAW_PATH,

        # Optional arguments
        sample_rate=1/1,
        coordinates="secondspectrum",
        only_alive=False)

    home_df, away_df = get_home_away_tracking(dataset)
    tracking_home = vel.calc_player_velocities(home_df, smoothing=True, filter_='moving_average')
    tracking_away = vel.calc_player_velocities(away_df, smoothing=True, filter_='moving_average')
    return tracking_home, tracking_away


tracking_home, tracking_away = load_data()
with open("../../notebooks/merged_df.pkl", 'rb') as f:
    full_merged = pickle.load(f)

st.title('City App')
h1 = full_merged[(full_merged.period_id_x == 2)]

period = st.slider("Threat by time:", min_value=1, max_value=h1.shape[0], value=(1, 1))
st.write(f"{h1.iloc[period[1]].minute}:{h1.iloc[period[1]].second}")
home = h1.iloc[period[0]-1:period[1]][h1.attacking_team == "away"]
grids = home.conrol_matrix.to_list()

st.write("Average threat per event in area")
features = []
for matrix in grids:
    features.append(matrix.flatten())
X = np.array(features)
wss = []
max_clusters = 6
for i in range(1, max_clusters):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    wss.append(kmeans.inertia_)

# find the elbow point using the KneeLocator package
kl = KneeLocator(range(1, max_clusters), wss, curve='convex', direction='decreasing')
k = kl.elbow
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
cluster_labels = kmeans.labels_
clustered_matrices = [[] for i in range(k)]
for i, matrix in enumerate(grids):
    clustered_matrices[cluster_labels[i]].append(matrix)

result = np.zeros(clustered_matrices[0][0].shape)
for matrix in clustered_matrices:
    result += np.mean(matrix, axis=0)

vmax = np.max(result)
fig1, ax = viz.plot_obso_grid(result, vmax)
st.pyplot(fig1)

st.write("Total cumultive threat")

cum_result = np.zeros(grids[0].shape)
for matrix in grids:
    cum_result += matrix
vmax = np.max(cum_result)

fig2, ax = viz.plot_obso_grid(cum_result, vmax)
st.pyplot(fig2)
# fig2, ax = viz.plot_obso_grid(avg_array2, vmax)

# col1, col2 = st.columns(2)
# with col1:
#     st.write("Home Team")
#     st.pyplot(fig1)
# with col2:
#     st.write("Away Team")
#     st.pyplot(fig2)

# fig, ax = plt.subplots(figsize=(12, 3))
# h1['OBSO'] = h1.conrol_matrix.apply(np.sum)
# h1['OBSO'] = h1['OBSO'] * h1['attacking_team'].apply(lambda x: 1 if x == 'home' else -1)
# # Create line chart with blue and red colors
# df = h1.reset_index(drop=True)
# plt.plot(df.index, df['OBSO'])
# plt.ylabel('OBSO')
# ax.axhline(y=0, linestyle='--')
# st.pyplot(fig)
# # Add labels to the chart

