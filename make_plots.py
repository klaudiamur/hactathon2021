#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 22:42:18 2021

@author: klaudiamur
"""


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.markers import MarkerStyle
import seaborn as sn
import math

path = '/Users/klaudiamur/Documents/hackathon/hactathon2021'
users_data = pd.read_csv(path+'/users_data.csv', index_col=0)

##colors
c_m = "#D6C9C3"
c_f = "#18144C"

## density plots for women/men
x = 'eigenvector_centrality_G1'   
data_tp_f = users_data[x][(users_data['gender_num'] == 1) & (users_data['heigth'] <20)]
data_tp_m = users_data[x][(users_data['gender_num'] == 0)& (users_data['heigth'] <20) ]

print(np.mean(data_tp_f))
print(np.mean(data_tp_m))

sn.displot(users_data[users_data['heigth'] < 10], x = x, hue="gender_num", kind = "kde")

np.mean(users_data[(users_data['core_number_G2']> 6) & (users_data['heigth'] > 4)]['gender_num'])

## find nodes that have good structural holes (good information flow)
users_data.nsmallest(10, columns= 'constraint_G1')[['Name',  'gender','div_hier_G1' ]]

### make network plots
def calc_avg_data(x, g):
    return np.mean(users_data[x][(users_data['gender_num'] == g)])

g = 1
nwn = 'G1'
nodes = np.arange(round(calc_avg_data('degree_centrality_'+nwn+'_tot', g))+1)
nodes = np.arange(10)
own_t = round(len(nodes)*calc_avg_data('homophiliy_'+nwn, g))
#clust = len(nodes)*calc_avg_data('clustering_'+nwn, g)
#own_t = round(len(nodes)*calc_avg_data('ratio_div_t_'+nwn, g)) ##0
#n_other_t = round(calc_avg_data('div_interaction_teams_'+nwn, g))
n_other_t = 1
n_p_o_t = math.ceil((len(nodes )- own_t)/n_other_t)
density =calc_avg_data('clustering_'+nwn, g)

c_dis = np.arange(0, n_other_t+2)/(n_other_t+2)

colors_own =[c_dis[0]]*(own_t+1)
cothers = [n_p_o_t*[c_dis[i]] for i in range(2, n_other_t+2)]
colors = colors_own + [item for sublist in cothers for item in sublist][:(len(nodes)-len(colors_own))]
colors = [c_f]*len(nodes)
edges = [(0, i) for i in nodes[1:]]
edge_c = [(i, j) for i in nodes for j in nodes  if (np.random.random() < density and i !=j )]
edges = edges + edge_c
Ge = nx.from_edgelist(edges)

#pos = nx.drawing.nx_agraph.graphviz_layout(Ge, prog = 'dot')
pos = {n:(math.cos(2*math.pi*k/(len(nodes)-1)), math.sin(k*2*math.pi/(len(nodes)-1))) for k, n in enumerate(nodes)}
pos[0] = (0,0)
#pos1 = {k:v for k, v in pos.items() if k in G_comm.node
fig = plt.figure(frameon=False, dpi=500, figsize=(5,5))

nx.draw_networkx_nodes(Ge, pos, 
                       #node_color='Purple', 
                       cmap = 'twilight_shifted',
                       vmax = 1.2,
                       vmin = -0.005,
                       node_color = colors,
                       node_size = 200,
                       #node_size = node_size,
                       alpha = 1
                       )
nx.draw_networkx_edges(Ge, pos, alpha = 0.4, edge_color='grey')


plt.show()

# =============================================================================
# plot pie charts of inclusivity
# =============================================================================

headcountt = {teamname:(np.mean(users_data['gender_num'][users_data['teamname']==teamname])) for teamname in pd.unique(users_data['teamname'])}

labels = ['Women', 'Men']


teamname = 20
#colors = cmap(np.arange(2))
colors = [c_f, c_m]
users_d = users_data[users_data['heigth'] < 3]
#users_d = users_data
headcount_t = np.mean(users_data['gender_num'])
headcount = np.mean(users_d['gender_num'])
ratio_f = np.mean(users_d['homophiliy_G1'][users_d['gender_num']==1])
ratio_m = np.mean(users_d['homophiliy_G1'][users_d['gender_num']==0])
ratio_f_t = np.mean(users_data[(users_data['gender_num']==1) ]['homophiliy_G1'])
ratio_m_t = np.mean(users_data['homophiliy_G1'][(users_data['gender_num']==0)])
ratio_f2 = np.mean(users_d['homophiliy_G2'][users_d['gender_num']==1])
ratio_m2 = np.mean(users_d['homophiliy_G2'][users_d['gender_num']==0])
ratio = np.mean(users_d['homophiliy_G_dir'])

#fig = plt.figure(figsize=(18, 10))
fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(2, 6)

#fig1, ax1 = plt.subplots(3, 2, figsize= (10, 18))

ax1 = plt.subplot(gs[0:2, 0:2]) #, figsize= (10,18))
ax1.pie([headcount_t,1- headcount_t] , labels = labels, colors =colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.set_title('Gender ratio of team')

#ax6 = plt.subplot(gs[0:2, 2:4]) #, figsize= (10,18))
#ax6.pie([headcount,1- headcount] , labels = labels, colors =colors, autopct='%1.1f%%',
 #       shadow=True, startangle=90)
#ax6.set_title('Gender ratio of management')

ax2 = plt.subplot(gs[0:2, 2:4])
ax2.pie([ratio_f_t,1- ratio_f_t] , labels = labels, colors =colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.pie(
ax2.set_title('Who women in the team interact with')

ax3 = plt.subplot(gs[0:2, 4:6])
ax3.pie([ratio_m_t,1- ratio_m_t] , labels = labels, colors =colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.pie(
ax3.set_title('Who men in the team interact with')

plt.show()
ax4 = plt.subplot(gs[0:2, 4:6])
ax4.pie([ratio_f,1- ratio_f] , labels = labels, colors =colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.pie(
ax4.set_title('Who female managers interact with')

ax5 = plt.subplot(gs[2:4, 4:6])
ax5.pie([ratio_m,1- ratio_m] , labels = labels, colors =colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.pie(
ax5.set_title('Who male managers interact with')


plt.show()
