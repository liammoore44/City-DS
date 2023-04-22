#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:10:58 2020
Module for visualising Metrica tracking and event data
Data can be found at: https://github.com/metrica-sports/sample-data
@author: Laurie Shaw (@EightyFivePoint)
"""

import matplotlib.pyplot as plt
import numpy as np
import src.pitch_control as pc


def swap_axes(line2d, xdata, ydata):
    line2d.set_xdata(ydata)
    line2d.set_ydata(xdata)


def plot_pitch(figax=None, field_dimen=(106.0, 68.0), field_color='green', linewidth=2, markersize=20, dpi=100, half=False, show_direction=False):
    """ plot_pitch
    
    Plots a soccer pitch. All distance units converted to meters.
    
    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    if figax is None:
        fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)  # create a figure 
    else:
        fig, ax = figax
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color == 'green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke'  # line color
        pc = 'w'  # 'spot' colors
    elif field_color == 'white':
        lc = 'k'
        pc = 'k'
    # ALL DIMENSIONS IN m
    border_dimen = (3, 3)  # include a border arround of the field of width 3m
    meters_per_yard = 0.9144  # unit conversion from yards to meters
    half_pitch_length = field_dimen[0]/2.   # length of half pitch
    half_pitch_width = field_dimen[1]/2.  # width of half pitch

    signs = [-1, 1]
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8*meters_per_yard
    box_width = 20*meters_per_yard
    box_length = 6*meters_per_yard
    area_width = 44*meters_per_yard
    area_length = 18*meters_per_yard
    penalty_spot = 12*meters_per_yard
    corner_radius = 1*meters_per_yard
    D_length = 8*meters_per_yard
    D_radius = 10*meters_per_yard
    D_pos = 12*meters_per_yard
    centre_circle_radius = 10*meters_per_yard
    # plot half way line # center circle
    ax.plot([0, 0], [-half_pitch_width, half_pitch_width], lc, linewidth=linewidth)
    ax.scatter(0.0, 0.0, marker='o', facecolor=lc, linewidth=0, s=markersize)
    y = np.linspace(-1, 1, 50)*centre_circle_radius
    x = np.sqrt(centre_circle_radius**2-y**2)
    ax.plot(x, y, lc, linewidth=linewidth)
    ax.plot(-x, y, lc, linewidth=linewidth)
    for s in signs:  # plots each line seperately
        # plot pitch boundary
        ax.plot([-half_pitch_length, half_pitch_length], [s*half_pitch_width, s*half_pitch_width], lc, linewidth=linewidth)
        ax.plot([s*half_pitch_length, s*half_pitch_length], [-half_pitch_width, half_pitch_width], lc, linewidth=linewidth)
        # goal posts & line
        ax.plot([s*half_pitch_length, s*half_pitch_length], [-goal_line_width/2., goal_line_width/2.], pc+'s', markersize=6*markersize/20., 
                linewidth=linewidth)
        # 6 yard box
        ax.plot([s*half_pitch_length, s*half_pitch_length-s*box_length], [box_width/2., box_width/2.], lc, linewidth=linewidth)
        ax.plot([s*half_pitch_length, s*half_pitch_length-s*box_length], [-box_width/2., -box_width/2.], lc, linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*box_length, s*half_pitch_length-s*box_length], [-box_width/2., box_width/2.], lc,
                linewidth=linewidth)
        # penalty area
        ax.plot([s*half_pitch_length, s*half_pitch_length-s*area_length], [area_width/2., area_width/2.], lc, linewidth=linewidth)
        ax.plot([s*half_pitch_length, s*half_pitch_length-s*area_length], [-area_width/2., -area_width/2.], lc, linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*area_length, s*half_pitch_length-s*area_length], [-area_width/2., area_width/2.], lc,
                linewidth=linewidth)
        # penalty spot
        ax.scatter(s*half_pitch_length-s*penalty_spot, 0.0, marker='o', facecolor=lc, linewidth=0, s=markersize)
        # corner flags
        y = np.linspace(0, 1, 50)*corner_radius
        x = np.sqrt(corner_radius**2-y**2)
        ax.plot(s*half_pitch_length-s*x, -half_pitch_width+y, lc, linewidth=linewidth)
        ax.plot(s*half_pitch_length-s*x, half_pitch_width-y, lc, linewidth=linewidth)
        # draw the D
        y = np.linspace(-1, 1, 50)*D_length  # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2-y**2)+D_pos
        ax.plot(s*half_pitch_length-s*x, y, lc, linewidth=linewidth)
    if (show_direction):
        ax.annotate("Direction of play", xy=(-2, -31), xytext=(-40, -31.5),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0]/2. + border_dimen[0]
    ymax = field_dimen[1]/2. + border_dimen[1]
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)

    if half is True:
        for line in ax.lines:
            swap_axes(line, line.get_xdata(), line.get_ydata())
        ax.set_xlim([-ymax, ymax])
        ax.set_ylim([-xmax, 0])

    return (fig, ax)


def plot_frame(hometeam, awayteam, figax=None, team_colors=('r', 'b'), field_dimen=(106.0, 68.0), 
               include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False, dpi=100):
    """ plot_frame( hometeam, awayteam )
    
    Plots a frame of Metrica tracking data (player positions and the ball) on a football pitch. All distances should be in meters.
    
    Parameters
    -----------
        hometeam: row (i.e. instant) of the home team tracking data frame
        awayteam: row of the away team tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot, 
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    if figax is None:  # create new pitch 
        fig, ax = plot_pitch(field_dimen=field_dimen, dpi=dpi)
    else:  # overlay on a previously generated pitch
        fig, ax = figax  # unpack tuple
    # plot home & away teams in order
    for team, color in zip([hometeam, awayteam], team_colors):
        x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c !='ball_x']  # column header for player x positions
        y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c !='ball_y']  # column header for player y positions
        ax.plot(team[x_columns], team[y_columns], color+'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha) # plot player positions
        if include_player_velocities:
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
            ax.quiver(team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,
                      width=0.0015, headlength=5, headwidth=3, alpha=PlayerAlpha)
        if annotate:
            [ax.text(team[x]+0.5, team[y]+0.5, x.split('_')[1], fontsize=10, color=color) for x, y in zip(x_columns, y_columns) if not
                (np.isnan(team[x]) or np.isnan(team[y]))] 
    # plot ball
    ax.plot(hometeam['ball_x'], hometeam['ball_y'], 'ko', color="yellow", markersize=6, alpha=1.0, linewidth=0)
    return fig, ax


def plot_frame_players(frame, tracking_home, tracking_away, attacking_team, grid, alpha=0.7, include_player_velocities=True,
                       annotate=True, field_dimen=(106., 68.,), n_grid_cells_x=50):

    fig, ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    plot_frame(tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig, ax), PlayerAlpha=alpha,
               include_player_velocities=include_player_velocities, annotate=annotate)
    
    xgrid = np.linspace(-field_dimen[0]/2., field_dimen[0]/2., 50)
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    ygrid = np.linspace(-field_dimen[1]/2., field_dimen[1]/2., n_grid_cells_y)
    im = ax.imshow(np.flipud(grid), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                   interpolation='hanning', vmin=0.0, vmax=np.max(grid), cmap="Spectral_r")
    # Add colorbar
    cbar = fig.colorbar(im)
    return fig, ax

def plot_players(frame, tracking_home, tracking_away, alpha=0.7, include_player_velocities=True,
                       annotate=True, field_dimen=(106., 68.,), n_grid_cells_x=50):

    fig, ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    plot_frame(tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig, ax), PlayerAlpha=alpha,
               include_player_velocities=include_player_velocities, annotate=annotate)
    
    return fig, ax


def plot_obso_grid(grid, vmax, tracking_home, period, attacking_team, field_dimen=(106., 68.,), n_grid_cells_x=50, show_direction=False):

    home_attack = pc.where_home_team_attacks(tracking_home)

    fig, ax = plot_pitch(field_color='white', field_dimen=field_dimen, show_direction=show_direction)
    xgrid = np.linspace(-field_dimen[0]/2., field_dimen[0]/2., 50)
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    ygrid = np.linspace(-field_dimen[1]/2., field_dimen[1]/2., n_grid_cells_y)
    # normalize to plot left to right
    if home_attack == -1:
        if (attacking_team == "home" and period == 1):

            grid = grid[:, ::-1]
        elif (attacking_team == "away" and period == 2):
            grid = grid[:, ::-1]
    else:
        if (attacking_team == "home" and period == 2):

            grid = grid[:, ::-1]
        elif (attacking_team == "away" and period == 1):
            grid = grid[:, ::-1]
    im = ax.imshow(np.flipud(grid), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                   interpolation='hanning', vmin=0.0, vmax=vmax, cmap="Spectral_r")
    # Add colorbar
    cbar = fig.colorbar(im)
    return fig, ax


def plot_pitch_control_for_frame(frame, tracking_home, tracking_away, attacking_team, params, alpha=0.7, include_player_velocities=True, 
                                 annotate=True, field_dimen=(106., 68.,), n_grid_cells_x=50):
    """ plot_pitch_control_for_frame(frame, tracking_home, tracking_away, params ,PPCF, xgrid, ygrid )
    
    Plots the pitch control surface at the instant of the frame. Player and ball positions are overlaid.
    
    Parameters
    -----------
        frame: the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
        xgrid: Positions of the pixels in the x-direction (field length) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
        ygrid: Positions of the pixels in the y-direction (field width) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
    Returns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """

    # plot frame and event
    fig, ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    plot_frame(tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig, ax), PlayerAlpha=alpha,
               include_player_velocities=include_player_velocities, annotate=annotate)
    
    # generate pitch control
    PPCF, xgrid, ygrid = pc.generate_pitch_control_for_frame(frame, tracking_home, tracking_away, attacking_team, params,
                                                             field_dimen=field_dimen, n_grid_cells_x=n_grid_cells_x)

    # plot pitch control surface
    if attacking_team == 'Home':
        cmap = 'bwr'
    else:
        cmap = 'bwr_r'

    im = ax.imshow(np.flipud(PPCF), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)), interpolation='hanning',
                   vmin=0.0, vmax=1.0, cmap=cmap, alpha=0.5)
    cbar = fig.colorbar(im)
    cbar.set_label('Pitch Control')
    return fig, ax


def plot_transition_proba_for_frame(frame, tracking_home, tracking_away, attacking_team, params, alpha = 0.7, include_player_velocities=True, annotate=True, field_dimen = (106.,68.,), n_grid_cells_x = 50):
    """ plot_transition_prob_for_frame(frame, tracking_home, tracking_away, events, params ,PPCF, xgrid, ygrid )
    
    Plots the transition probability surface at the instant of the frame. Player and ball positions are overlaid.
    
    Parameters
    -----------
        frame: the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        xgrid: Positions of the pixels in the x-direction (field length) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
        ygrid: Positions of the pixels in the y-direction (field width) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
    Returns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """    
    # plot frame and event
    fig, ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    plot_frame(tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig, ax), PlayerAlpha=alpha,
               include_player_velocities=include_player_velocities, annotate=annotate)

    # generate pitch control
    PPCF, xgrid, ygrid, T = pc.generate_transition_probability_for_frame(frame, tracking_home, tracking_away, attacking_team, params,
                                                                          field_dimen=field_dimen, n_grid_cells_x=n_grid_cells_x)

    # plot pitch control surface
    if attacking_team == 'Home':
        cmap = 'Reds'
    else:
        cmap = 'Blues'

    im = ax.imshow(np.flipud(T), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),interpolation='hanning',vmin=0.0,cmap=cmap)
    cbar = fig.colorbar(im)
    cbar.set_label('Transitional probability')
    return fig, ax


def generate_relevant_pitch_for_frame(frame, tracking_home, tracking_away, attacking_team, params, field_dimen = (106.,68.,), n_grid_cells_x = 50):
    """ generate_relevant_pitch_for_frame
    
    Evaluates relevant pitch for frame surface over the entire field at the moment of the given frame
    
    Parameters
    -----------
        frame: instant at which the transition surface should be calculated
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        events: Dataframe containing the event data
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        
    Returns
    -----------
        rel_PPCF : relevant pitch surface
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)
    """
    
    PPCFa, xgrid, ygrid, T = pc.generate_transition_probability_for_frame(frame, tracking_home, tracking_away, attacking_team, params,
                                                                          field_dimen=field_dimen, n_grid_cells_x=n_grid_cells_x)
    rel_PPCF = PPCFa*T
    
    return rel_PPCF, xgrid, ygrid


def plot_relevant_pitch_for_frame(frame, tracking_home, tracking_away, attacking_team, params, alpha=0.7, include_player_velocities=True, 
                                  annotate=True, field_dimen=(106., 68.,), n_grid_cells_x=50):
    """ plot_relevant_pitch_for_frame(frame, tracking_home, tracking_away, events, params ,PPCF, xgrid, ygrid )

    Plots the relevant pitch surface at the instant of the frame. Player and ball positions are overlaid.

    Parameters
    -----------
        frame: the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        xgrid: Positions of the pixels in the x-direction (field length) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
        ygrid: Positions of the pixels in the y-direction (field width) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)

    Returns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """    
    # plot frame and event
    fig, ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    plot_frame(tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig, ax), PlayerAlpha=alpha,
               include_player_velocities=include_player_velocities, annotate=annotate)

    # generate pitch control
    rel_PPCF, xgrid, ygrid = generate_relevant_pitch_for_frame(frame, tracking_home, tracking_away, attacking_team, params,
                                                               field_dimen=field_dimen, n_grid_cells_x=n_grid_cells_x)

    # find attacking team

    # plot relevant pitch control surface
    if attacking_team == 'Home':
        cmap = 'Reds'
    else:
        cmap = 'Blues'
    im = ax.imshow(np.flipud(rel_PPCF), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)), interpolation='hanning',
                   vmin=0.0, cmap=cmap, alpha=1)
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label('Relavant pitch control probability')
    return fig, ax


def plot_expected_goals_surface_for_frame(frame, tracking_home, tracking_away, attacking_team, params, alpha=0.7,
                                          include_player_velocities=True, annotate=True, field_dimen=(106., 68.,),
                                          n_grid_cells_x=50):

    fig, ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    plot_frame(tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig, ax), PlayerAlpha=alpha,
               include_player_velocities=include_player_velocities, annotate=annotate)

    xT, xgrid, ygrid = pc.generate_expected_threat_surface(attacking_team, tracking_home, frame, field_dimen=field_dimen,
                                                           n_grid_cells_x=n_grid_cells_x)

    xmin = np.amin(xgrid)
    xmax = np.amax(xgrid)

    im = ax.imshow(np.flipud(xT), extent=(xmin, xmax, np.amin(ygrid), np.amax(ygrid)), interpolation='hanning', vmin=0.0,
                   vmax=1.0, cmap='Spectral_r', alpha=1)
    print(xT.max())
    print(xT.min())

    cbar = fig.colorbar(im)
    cbar.set_label('XT')

    return (fig, ax)


def generate_off_ball_scoring_opportunity_for_frame(frame, tracking_home, tracking_away, attacking_team, params,
                                                    field_dimen=(106., 68.,), n_grid_cells_x=50):

    xT, xgrid, ygrid = pc.generate_expected_threat_surface(attacking_team, tracking_home, frame, field_dimen=field_dimen,
                                                           n_grid_cells_x=n_grid_cells_x)
    rel_PPCF, xgrid, ygrid = generate_relevant_pitch_for_frame(frame, tracking_home, tracking_away, attacking_team, params,
                                                               field_dimen=field_dimen, n_grid_cells_x=n_grid_cells_x)

    return (xT*rel_PPCF, xgrid, ygrid)


def plot_scoring_opp_for_frame(frame, tracking_home, tracking_away, attacking_team, params,
                               alpha=0.7, include_player_velocities=True, annotate=True, field_dimen=(106., 68.,), n_grid_cells_x=50):
    
    fig, ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    plot_frame(tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig, ax), PlayerAlpha=alpha,
               include_player_velocities=include_player_velocities, annotate=annotate)
    
    off_scoring, xgrid, ygrid = generate_off_ball_scoring_opportunity_for_frame(frame, tracking_home, tracking_away, attacking_team, params,
                                                                                field_dimen=field_dimen, n_grid_cells_x=n_grid_cells_x)

    # plot pitch control surface
    if attacking_team == 'Home':
        cmap = 'Reds'
    else:
        cmap = 'Blues'

    im = ax.imshow(np.flipud(off_scoring), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),interpolation='hanning',
                   vmin=0.0, vmax=np.max(off_scoring), cmap=cmap)
    # Add colorbar
    cbar = fig.colorbar(im)
    print('off ball expected threat: '+str(round(np.sum(off_scoring)*100, 1)) + "%")
    
    return (fig, ax)
