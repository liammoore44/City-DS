import pandas as pd
import numpy as np

def find_changes(row, df, second, threshold):
    vals = df.loc[((df.time - row.time) <= second) & ((row.time - df.time) < 0) & (df.period_id_x==row.period_id_x)]
    vals['obso'] = np.where(vals.team_id==row.team_id, vals.obso, -vals.obso)
    vals['current_obso'] = row.obso
    vals['change'] = (vals.obso - vals.current_obso) / vals.current_obso * 100
    over_change = vals.loc[vals.change >= threshold]
    over_raw = vals.loc[vals['obso'] >= 0.015]
    if len(over_change) > 0 and len(over_raw) > 0:
        return True
    return False

def find_obso_swings(df, second, threshold, half, team):
    df['obso'] = df.conrol_matrix.apply(lambda x : np.sum(x))
    df['high_impact'] = df.apply(lambda x: find_changes(x, df, second, threshold), axis=1)
    impact_df = df.loc[(df.high_impact==True) & (df.period_id_x == half) & (df.attacking_team == team)]
    impact_df['next_time_diff'] = impact_df.time.shift(-1) - impact_df.time
    impact_df = impact_df.loc[(impact_df.next_time_diff > 3) & (impact_df.next_time_diff > 0)]
    result = []
    for i, row in impact_df.iterrows():
        res = df.loc[(df.period_id_x==row.period_id_x) & (df.time >= (row.time - (second*2.5))) & (df.time <= (row.time + (second*2.5)))]
        res['OBSO (%)'] = round(res.obso * 100, 2)
        result.append(res)
    impact_df['OBSO (%)'] = round(impact_df.obso * 100, 2)
    return result, impact_df