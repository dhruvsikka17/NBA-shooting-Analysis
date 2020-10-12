import pandas as pd
import numpy as np
from nba_api.stats import endpoints
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams,players
teams=teams.get_teams()
nba_players = players.get_players()

from nba_api.stats.library.parameters import SeasonAll
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
import json
import requests
import streamlit as st

st.title("NBA Player Analysis Web App")
st.sidebar.title("User Inputs")
st.write("""Shown Below are the Data charts for Player Performance Analysis.""")

team=pd.DataFrame(teams)
team_name = st.sidebar.selectbox("Team's Name",team.full_name)
team_id= [team_id for team_id in teams
             if team_id['full_name'] == team_name][0]['id']

player_name= st.sidebar.text_input("Player's Full name",value="Trae Young")
player_id= [player_id for player_id in nba_players
             if player_id['full_name'] == player_name][0]['id']

season = st.sidebar.text_input("Season",value="2018-19")


st.header('Player Stats')
data = endpoints.leagueleaders.LeagueLeaders(season=season)
df = data.league_leaders.get_data_frame()
df['PPG']=df.PTS/df.GP
df['RPG']=df.REB/df.GP
df['APG']=df.AST/df.GP
df['SPG']=df.STL/df.GP
df['BPG']=df.BLK/df.GP
df['TPG']=df.TOV/df.GP
df['FG%']=df.FG_PCT
df['FG3%']=df.FG3_PCT
df['FT%']=df.FT_PCT
df =df.drop(columns=['FG_PCT','FG3_PCT','FT_PCT','GP','PLAYER_ID','RANK','TEAM','MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','PTS','REB','AST','STL','BLK','TOV','PF','AST_TOV','STL_TOV'])
player_stats = df.loc[df['PLAYER'] == player_name]
st.table(player_stats)

shot_json = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id = player_id,
                context_measure_simple = 'PTS',
                season_nullable = season,
                season_type_all_star = 'Regular Season')
shot_data = json.loads(shot_json.get_json())

relevant_data = shot_data['resultSets'][0]
headers = relevant_data['headers']
rows = relevant_data['rowSet']
player_data = pd.DataFrame(rows)
player_data.columns = headers

def draw_court(ax=None, color='black', lw=2, outer_lines=False):

        if ax is None:
            ax = plt.gca()

        hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

        backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

        outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                              fill=False)

        inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                              fill=False)

        top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                             linewidth=lw, color=color, fill=False)

        bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                                linewidth=lw, color=color, linestyle='dashed')

        restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                         color=color)

        corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                                   color=color)
        corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
        three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                        color=color)

        center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                               linewidth=lw, color=color)
        center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                               linewidth=lw, color=color)

        court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                          bottom_free_throw, restricted, corner_three_a,
                          corner_three_b, three_arc, center_outer_arc,
                          center_inner_arc]

        if outer_lines:

            outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                    color=color, fill=False)
            court_elements.append(outer_lines)

        for element in court_elements:
            ax.add_patch(element)

        return ax

st.header("Shot Chart")
no_clusters= st.slider("No. Of Clusters",1,10)
playerkmeans=player_data[['LOC_X','LOC_Y']]
km_res=KMeans(n_clusters=no_clusters).fit(playerkmeans)
y_kmeans = km_res.predict(playerkmeans)

fig = plt.figure(figsize=(15,13))
plt.scatter(data=playerkmeans,x='LOC_X',y='LOC_Y',c=y_kmeans,s=100,cmap='rainbow')
draw_court(outer_lines=True)
plt.xlim(-300,300)
plt.ylim(-80,450)
st.pyplot(fig)
