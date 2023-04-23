import streamlit as st
import pickle
from kloppy import secondspectrum
import sys
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from mplsoccer import Pitch

import src.visualizations as viz
from src.second_spectrum_utils import get_home_away_tracking
import src.player_velocities as vel

st.set_page_config(layout='wide')


@st.cache_data
def load_data(match):
    if match == "Man City v Liverpool":
        TRACKING_HOME = "src/ta_ManCity_Liverpool.pkl"
        TRACKING_AWAY = "src/th_ManCity_Liverpool.pkl"
        EVENT_NAME = "src/ManCity_Liverpool.pkl"
    elif match == "Man City v Tottenham":
        TRACKING_HOME = "src/ta_ManCity_Tottenham.pkl"
        TRACKING_AWAY = "src/th_ManCity_Tottenham.pkl"
        EVENT_NAME = "src/ManCity_Tottenham.pkl"
    elif match == "Man City v Arsenal":
        TRACKING_HOME = "src/ta_ManCity_Arsenal.pkl"
        TRACKING_AWAY = "src/th_ManCity_Arsenal.pkl"
        EVENT_NAME = "src/ManCity_Arsenal.pkl"
    elif match == "Man City v Brighton":
        TRACKING_HOME = "src/ta_ManCity_Brighton.pkl"
        TRACKING_AWAY = "src/th_ManCity_Brighton.pkl"
        EVENT_NAME = "src/ManCity_Brighton.pkl"
    elif match == "Man City v Leicester":
        TRACKING_HOME = "src/ta_ManCity_Leicester.pkl"
        TRACKING_AWAY = "src/th_ManCity_Leicester.pkl"
        EVENT_NAME = "src/ManCity_Leicester.pkl"

    with open(TRACKING_HOME, 'rb') as f:
        tracking_home = pickle.load(f)
    with open(TRACKING_AWAY, 'rb') as f:
        tracking_away = pickle.load(f)
    with open(EVENT_NAME, 'rb') as f:
        full_merged = pickle.load(f)
    return tracking_home, tracking_away, full_merged


st.title('City App')

match = st.selectbox("Select Match", ("Man City v Liverpool",
                                      "Man City v Tottenham",
                                      "Man City v Arsenal",
                                      "Man City v Brighton",
                                      "Man City v Leicester"))

tracking_home, tracking_away, full_merged = load_data(match)

col1, col2 = st.columns(2)
with col1:
    team = st.selectbox(
        'Team',
        ('home', 'away'))
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        half = st.radio(
            "Half",
            (1, 2))
    with subcol2:
        placeholder_clusters = st.empty()

    h1 = full_merged[(full_merged.period_id_x == half) & (full_merged.attacking_team == team)]
    period = st.slider("Threat by time:", min_value=1, max_value=h1.shape[0], value=(0, 10))

    if (period[1] - period[0] < 6):
        st.error('Please select a minimum range of 6 events', icon="🚨")
    else:
        st.write(f"Match Time: {h1.iloc[period[0]].minute}:{h1.iloc[period[0]].second}-{h1.iloc[period[1]-1].minute}:{h1.iloc[period[1]-1].second}")
        df = h1.iloc[period[0]:period[1]-1].copy()
        fig, ax = plt.subplots(figsize=(12, 3))
        df['OBSO'] = df.conrol_matrix.apply(np.sum)
        # Create line chart with blue and red colors
        x = df.reset_index(drop=True)
        plt.bar(x.index, x['OBSO'])
        plt.ylabel('OBSO')
        plt.xlim(xmin=period[0], xmax=h1.shape[0])
        ax.axhline(y=0, linestyle='--')
        st.pyplot(fig)

        with col2:
            grids = df.conrol_matrix.to_list()
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
            k = kl.elbow if kl.elbow is not None else 2
            with placeholder_clusters.container():
                clusters = ["All Zones"]
                for i in range(k):
                    clusters.append(f"Zone {i+1}")
                cluster = st.radio("Zone", clusters)

            kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
            cluster_labels = kmeans.labels_
            clustered_matrices = [[] for i in range(k)]
            for i, matrix in enumerate(grids):
                clustered_matrices[cluster_labels[i]].append(matrix)

            cluster_grids = []
            result = np.zeros(clustered_matrices[0][0].shape)
            for matrix in clustered_matrices:
                avg_arr = np.mean(matrix, axis=0)
                cluster_grids.append(avg_arr)
                result += avg_arr

            vmax = np.max(result)
            if cluster == "All Zones":
                result = result
            else:
                mask = cluster_labels == int(cluster.split(" ")[1])-1
                df = df[mask]
                result = cluster_grids[int(cluster.split(" ")[1])-1]
            fig1, ax = viz.plot_obso_grid(result, vmax, tracking_home, half, team, show_direction=True)
            st.pyplot(fig1)

            pitch = Pitch()
            # specifying figure size (width, height)
            fig2, ax = pitch.draw(figsize=(8, 4))
            passes = df[(df.type_name == "Pass") & ~(df.pass_outcome_id.isin([9, 75]))]
            passes['x'] = passes.location.apply(lambda x: x[0])
            passes['y'] = passes.location.apply(lambda x: x[1])
            passes['x_dest'] = passes.pass_end_location.apply(lambda x: x[0])
            passes['y_dest'] = passes.pass_end_location.apply(lambda x: x[1])
            pitch.arrows(passes.x, passes.y, passes.x_dest, passes.y_dest, color="blue", lw=0.1, width=1, ax=ax)
            st.write("Completed passes during selected time-zone combination")
            st.pyplot(fig2)

    player_values_df = df[['player_name', 'position_name', 'OBSO', 'obv_total_net']]
    grouped_vals = player_values_df.groupby(['player_name', 'position_name']).sum()
    grouped_vals.rename(columns={"OBSO": "Off Ball Scoring Opp", "obv_total_net": "On Ball Value"}, inplace=True)
    grouped_vals = grouped_vals.sort_values(by=["On Ball Value"], ascending=False)
    style = grouped_vals.style.background_gradient(cmap='RdYlGn')

    st.table(style)
    # Add labels to the chart

    # st.write("Total cumultive threat")

    # cum_result = np.zeros(grids[0].shape)
    # for matrix in grids:
    #     cum_result += matrix
    # vmax = np.max(cum_result)

    # fig2, ax = viz.plot_obso_grid(cum_result, vmax)
    # st.pyplot(fig2)
