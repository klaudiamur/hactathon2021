#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 21:00:13 2021

@author: klaudiamur
"""

import pandas as pd
import numpy as np
import networkx as nx
import datetime
import bisect
import gender_guesser.detector as gender
import matplotlib.pyplot as plt

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return a[i - 1]
    raise ValueError


def make_time_network(users, df,
                      rec_internal):  ### this is the directed network of emails sent! i is sender, j is recipient
    time_min = min(df['ts']).date()
    time_max = max(df['ts']).date()

    d = +datetime.timedelta(days=1)

    t = time_min

    timescale = []
    while t <= time_max:
        t = t + d
        timescale.append(t)

    duration = len(timescale)

    network = np.zeros((len(users), len(users), duration), dtype=int)    

    for k in range(len(df)):
        sen = df.iloc[k]['sender']
        rec = df.iloc[k]['recipient']

        if sen in rec_internal:
            if rec in rec_internal:
    
                time = df.iloc[k]['ts'].date()
    
                t = (time - time_min).days
    
                sen_indx = users.index(sen)
                rec_indx = users.index(rec)
    
                network[sen_indx, rec_indx, t] += 1

    for i in range(len(users)):
        network[i][i] = 0

    return {'nw': network, 'ts': timescale}


def modify_time_network(network, ts, timescale):
    ### make weekly timescale network out of daily timescale network

    if timescale == 'week':
        time_ts = [i.isocalendar()[1] for i in ts]
    elif timescale == 'month':
        time_ts = [i.month for i in ts]
    else:
        raise ValueError('The timescale has to be week or month')

    u, indices = np.unique(time_ts, return_inverse=True)
    ### sum up the network based on the number in weekly_ts!
    time_network = np.zeros((np.shape(network)[0], np.shape(network)[1], len(u)))

    for i in list(range(len(u))):
        indx = np.nonzero(i == indices)[0]
        new_network = np.sum(network[:, :, indx], axis=2)
        time_network[:, :, i] = new_network
        # weekly_network = np.stack(weekly_network, new_network)

    return {'nw': time_network, 'ts': u}


def make_tight_nw(dir_matrix, biwe, wts, tst):
    nw_tot = np.sum(dir_matrix['nw'], axis=2)
    ts = dir_matrix['ts']
    ts_bw = ts[0::biwe] + [ts[-1]]

    nw_biwe = np.zeros((len(dir_matrix['nw']), len(dir_matrix['nw']), len(ts_bw)))

    for i in range(len(ts_bw)):
        t0 = i * biwe
        t1 = t0 + biwe
        if t1 > len(ts):
            t1 = len(ts)
        nw_tmp = np.sum(dir_matrix['nw'][:, :, t0:t1], axis=2)
        nw_biwe[:, :, i] = nw_tmp

    nw_biwe_tot = [[1 if (np.count_nonzero(nw_biwe[i, j]) >= wts and nw_tot[i, j] >= tst and nw_tot[
        j, i] >= tst and np.count_nonzero(nw_biwe[j, i]) >= wts) else 0 for i in range(len(nw_biwe))] for j in
                   range(len(nw_biwe))]

    G = nx.from_numpy_matrix(np.array(nw_biwe_tot))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G, nw_biwe_tot


path = '/Users/klaudiamur/Documents/hackathon'

data_tmp = pd.read_csv(path+'/emails.csv')

senders = {}
recipients = {}
ts = {}
ts_or = {}
j = 0

for i in range(len(data_tmp)):
    message1 = data_tmp['message'][i].partition('From:')[-1]
    sender = message1.split()[0]
    sender = sender.replace(',', '')
    message1 = data_tmp['message'][i].partition('To:')[-1]
    recipient = message1.split()[0]
    recipient = recipient.replace(',', '')
    message1 = data_tmp['message'][i].partition('Date: ')[-1]
    t = pd.to_datetime(message1.partition('\n')[0], utc=True) # split at new line
    if t > pd.to_datetime('1999-01-01', utc=True) and t < pd.to_datetime('2002-01-01', utc=True):
        senders[j] = sender
        recipients[j] = recipient
        ts[j] = t
        ts_or[j] = pd.to_datetime(message1.partition('\n')[0])
        j += 1

data = pd.DataFrame.from_dict(senders, orient='index', columns= ['sender'])
data['recipient'] = pd.Series(recipients)
data['ts'] = pd.Series(ts)
data['ts_or'] = pd.Series(ts_or)
#data = pd.DataFrame(index=range(len(data_tmp)), columns = ['sender', 'recipient', 'ts'])
### find from, to and date in message

    
users_list_all = [i for i in senders.values() if 'enron' in i]+[i for i in recipients.values() if 'enron' in i]

users_list, count = np.unique(users_list_all, return_counts=True)
count_sort_ind = np.argsort(-count)
users_list = users_list[count_sort_ind]
email = {i:n for i, n in enumerate(users_list)}
first_name = {i:n.split('.')[0].capitalize() for i, n in enumerate(users_list)}

users_data_tmp = pd.DataFrame.from_dict(first_name, orient='index', columns = ['first_name'])
users_data_tmp['Email'] = pd.Series(email)
### get gender
d = gender.Detector()
genderlist = {}
for n in users_data_tmp.index:
    fn = users_data_tmp['first_name'][n]

    gender1 = d.get_gender(fn)
    genderlist[n] = gender1

users_data_tmp['gender'] = pd.Series(genderlist)
dict1 = {}
for n in users_data_tmp.index:
    g = users_data_tmp['gender'][n]
    # print(g)
    if g in ['female', 'mostly_female']:
        dict1[n] = 1
        # print(1)
    elif g in ['male', 'mostly_male']:
        dict1[n] = 0
        # print(0)
users_data_tmp['gender_num'] = pd.Series(dict1)

### pick only the ones that have a first name!
d = users_data_tmp['gender_num'].to_dict()
human_users_indx = [i for i, n in genderlist.items() if n != 'unknown'] ### check so that mostly_female becomes 1
human_users = [email[i] for i in human_users_indx]   
users_data =  pd.DataFrame.from_dict({i:n for i, n in enumerate(human_users)}, orient='index', columns= ['Email'])
users_data['first_name'] = pd.Series({i:users_data_tmp['first_name'][n] for i, n in enumerate(human_users_indx)})
users_data['gender'] = pd.Series({i:users_data_tmp['gender'][n] for i, n in enumerate(human_users_indx)})
users_data['gender_num'] = pd.Series({i:users_data_tmp['gender_num'][n] for i, n in enumerate(human_users_indx)})
## copy all the ones from the temporary dataframe (how?)
## make df just with human users!!

size = 500    
dir_matrix_listed = make_time_network(human_users[:size], data, human_users[:size])  

nw_tot = np.sum(dir_matrix_listed['nw'], axis=2)

G_dir = nx.from_numpy_matrix(nw_tot, create_using=nx.DiGraph)
emails_to_themselves_or_no_emails = list(nx.isolates(G_dir))
G_dir.remove_nodes_from(list(nx.isolates(G_dir)))

G1, G1_matrix = make_tight_nw(dir_matrix_listed, 7, 1, 5)
G2, G2_matrix = make_tight_nw(dir_matrix_listed, 14, 12, 2)
  
out_send_stats_tot = np.sum(nw_tot, axis=1)
in_send_stats_tot = np.sum(nw_tot, axis=0)
out_send_stats_dic = {i: out_send_stats_tot[i] for i in range(len(out_send_stats_tot))}
in_send_stats_dic = {i: in_send_stats_tot[i] for i in range(len(in_send_stats_tot))}

# out_send_stats = {i:out_send_stats_tot[i] for i in G_dir.nodes}
# in_send_stats = {i:in_send_stats_tot[i] for i in G_dir.nodes} ##wrong node index!!

out_degree = dict(G_dir.out_degree)
in_degree = dict(G_dir.in_degree)
                 
## get the name!                 
users_data['tot_msg_sent'] = pd.Series(out_send_stats_dic)
users_data['tot_msg_recieved'] = pd.Series(in_send_stats_dic)
users_data['out_degree'] = pd.Series(out_degree)
users_data['in_degree'] = pd.Series(in_degree)



networks = {'G1': G1, 'G2': G2}

list_of_methods = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality',
                   'clustering', 'core_number', 'node_clique_number', 'constraint',
                   # 'eccentricity']
                   ]
for name, ntwork in networks.items():

    for i in list_of_methods:
        method_to_call = getattr(nx, i)
        result = method_to_call(ntwork)
        users_data[i + '_' + name] = pd.Series(result)


users_data['degree_centrality_G1_tot'] = users_data['degree_centrality_G1'] * (len(G1.nodes) - 1)
users_data['degree_centrality_G2_tot'] = users_data['degree_centrality_G2'] * (len(G2.nodes) - 1)

# =============================================================================
# Identify burnout risk
# =============================================================================

over_work = pd.DataFrame(users_data['Email'])
work_days = np.zeros(len(over_work), dtype=int)
weekends = np.zeros(len(over_work), dtype=int)
after_wh = np.zeros(len(over_work), dtype=int)

df_dic = data.to_dict('records')
for row in df_dic: #use their local time?
    sen = row['sender']    
    if sen in human_users:
        indx0 = users_data[users_data['Email'] == sen].index.values
        if len(indx0) > 0:
            indx = indx0[0]
            # rec = [i for i in rec if i in rec_internal]
            #t = data.iloc[k]['ts_or']
            t = row['ts_or']
            #if users_data['Location'][indx] in offsetdic.keys():
            #    t_loc = t + offsetdic[users_data['Location'][indx]]
            #else:
            #    t_loc = t
            day = t.date().weekday()
            hour = t.hour
            #day = df.iloc[k]['origin_timestamp_utc'].date().weekday()
            #hour = 
            if day > 5: 
                weekends[indx] += 1
            elif hour < 7 or hour > 17:
                after_wh[indx] += 1
            else:
                work_days[indx] += 1

over_work['workdays'] = work_days
over_work['weekends'] = weekends
over_work['after_wh'] = after_wh
users_data['overwork_ratio_1'] = over_work['weekends'] / (
            over_work['workdays'] + over_work['weekends'])  ### only if they sent at least 10 emails!
users_data['overwork_ratio_1_ah'] = over_work['after_wh'] / (
            over_work['workdays'] + over_work['after_wh'])  ### only if they sent at least 10 emails!

users_data['overwork_tot_1'] = over_work['weekends']

users_data.loc[users_data['tot_msg_sent'] < 10, 'overwork_ratio_1'] = 0





G = G1
node_colors = ['blue' if users_data['gender_num'][k] == 0 else 'red' for k in G.nodes]
pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
#pos1 = {k:v for k, v in pos.items() if k in G_comm.nodes}

fig = plt.figure(frameon=False, dpi=500, figsize=(10,10))

nx.draw_networkx_nodes(G, pos, 
                       #node_color='Purple', 
                      # cmap = 'Purples',
                       #vmax = 1.2,
                       #vmin = -0.005,
                       node_color = node_colors,
                       #node_size = 1000,
                       #node_size = node_size,
                       alpha = 0.9
                       )
nx.draw_networkx_edges(G, pos, alpha = 0.4)

    #nx.draw_networkx_labels(G, pos)

plt.show()

# =============================================================================
# get homophoily (average of gender_num for neighbours)
# =============================================================================

#networks to look at: G
networks = {'G_dir': G_dir, 'G1': G1, 'G2': G2}
G = G1

for name, G in networks.items():
    results_dict = {}
    for n in G.nodes():
        gender_ratio = np.mean([users_data['gender_num'][n1] for n1 in G.neighbors(n)])
        results_dict[n] = gender_ratio
        
    users_data['homophiliy_'+name] = pd.Series(results_dict)    
        
users_data.to_csv('users_data.csv')
### make a simple pie chart
labels = ['Women', 'Men']
headcount = np.mean(users_data['gender_num'][:size])

ratio_f = np.mean(users_data['homophiliy_G_dir'][users_data['gender_num']==1])
ratio_m = np.mean(users_data['homophiliy_G_dir'][users_data['gender_num']==0])
ratio_f2 = np.mean(users_data['homophiliy_G2'][users_data['gender_num']==1])
ratio_m2 = np.mean(users_data['homophiliy_G2'][users_data['gender_num']==0])
ratio = np.mean(users_data['homophiliy_G_dir'])
fig1, ax1 = plt.subplots(3, 2, figsize= (10, 18))

ax1[0, 0].pie([headcount,1- headcount] , labels = labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.pie(
ax1[0, 0].set_title('Actual gender ratio of company')
ax1[1, 0].pie([ratio_f,1- ratio_f] , labels = labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.pie(
ax1[1, 0].set_title('Who women interact with')
ax1[1, 1].pie([ratio_m,1- ratio_m] , labels = labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.pie(
ax1[1, 1].set_title('Who men interact with')
ax1[2, 0].pie([ratio_f2,1- ratio_f2] , labels = labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.pie(
ax1[2, 0].set_title('Who women closely collaborate with')
ax1[2, 1].pie([ratio_m2,1- ratio_m2] , labels = labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.pie(
ax1[2, 1].set_title('Who men closely collaborate with')

ax1[0, 1].axis('off')

#ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



