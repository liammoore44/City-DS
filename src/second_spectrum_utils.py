"""
Module for cleaning the raw second spectrum data to be used
with pitch control modules.
"""


def get_home_away_tracking(dataset):

    df = dataset.to_df()
    df.drop(columns=df.filter(regex='^(.*)(_d|_s)$').columns, inplace=True)
    home, away = dataset.metadata.teams
    columns_to_keep_home = [col for col in df.columns if
                            any(s in col for s in
                                [p.player_id for p in home.players])]
    columns_to_keep_away = [col for col in df.columns if
                            any(s in col for s in
                                [p.player_id for p in away.players])]

    columns_to_keep_home.extend(["period_id", "timestamp", "ball_x", "ball_y",
                                 "frame_id"])
    columns_to_keep_away.extend(["period_id", "timestamp", "ball_x", "ball_y",
                                 "frame_id"])
    home_df = df[columns_to_keep_home].copy()
    away_df = df[columns_to_keep_away].copy()

    away_player_rename = {**{f"{player.player_id}_y": f"{player.full_name}_y"
                             for player in away.players},
                          **{f"{player.player_id}_x": f"{player.full_name}_x"
                             for player in away.players}}
    home_player_rename = {**{f"{player.player_id}_y": f"{player.full_name}_y"
                             for player in home.players},
                          **{f"{player.player_id}_x": f"{player.full_name}_x"
                             for player in home.players}}
    home_df.rename(columns=home_player_rename, inplace=True)
    away_df.rename(columns=away_player_rename, inplace=True)
    home_df = home_df.rename(columns=lambda x: x.capitalize())
    away_df = away_df.rename(columns=lambda x: x.capitalize())
    home_df.rename(columns={"Timestamp": "Time [s]", "Period_id": "Period",
                            "Ball_x": "ball_x", "Ball_y": "ball_y"},
                   inplace=True)
    away_df.rename(columns={"Timestamp": "Time [s]", "Period_id": "Period",
                            "Ball_x": "ball_x", "Ball_y": "ball_y"},
                   inplace=True)
    home_df = home_df.set_index('Frame_id')
    away_df = away_df.set_index('Frame_id')

    return home_df, away_df
