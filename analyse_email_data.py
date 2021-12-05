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
        sen = df.iloc[k]['sender'].split("@", 1)[0]
        rec = [i.split("@", 1)[0] for i in df.iloc[k]['recipient']]

        if sen in rec_internal:
            rec = [i for i in rec if i in rec_internal]
            time = df.iloc[k]['ts'].date()

            t = (time - time_min).days

            sen_indx = users.index(sen)
            rec_indx = [users.index(r) for r in rec]

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

for i in range(len(data_tmp)):
    message1 = data_tmp['message'][i].partition('From:')[-1]
    sender = message1.split()[0]
    sender.replace(',', '')
    message1 = data_tmp['message'][i].partition('To:')[-1]
    recipient = message1.split()[0]
    recipient.replace(',', '')
    message1 = data_tmp['message'][i].partition('Date: ')[-1]
    t = pd.to_datetime(message1.partition('\n')[0], utc=True) # split at new line
    
    senders[i] = sender
    recipients[i] = recipient
    ts[i] = t

data = pd.DataFrame.from_dict(senders, orient='index', columns= ['sender'])
data['recipient'] = pd.Series(recipients)
data['ts'] = pd.Series(ts)
#data = pd.DataFrame(index=range(len(data_tmp)), columns = ['sender', 'recipient', 'ts'])
### find from, to and date in message

    
users_list = [i for i in senders.values() if 'enron' in i]+[i for i in recipients.values() if 'enron' in i]

users_list = np.unique(users_list)
first_name = {i:n.split('.')[0].capitalize() for i, n in enumerate(users_list)}

    
    
dir_matrix_listed = make_time_network(users_list, data, users_list)  

nw_tot = np.sum(dir_matrix_listed['nw'], axis=2)

G_dir = nx.from_numpy_matrix(nw_tot, create_using=nx.DiGraph)
emails_to_themselves_or_no_emails = list(nx.isolates(G_dir))
G_dir.remove_nodes_from(list(nx.isolates(G_dir)))

G1, G1_matrix = make_tight_nw(dir_matrix_listed, 7, 1, 2)
G2, G2_matrix = make_tight_nw(dir_matrix_listed, 7, 9, 2)
  
out_send_stats_tot = np.sum(nw_tot, axis=1)
in_send_stats_tot = np.sum(nw_tot, axis=0)
out_send_stats_dic = {i: out_send_stats_tot[i] for i in range(len(out_send_stats_tot))}
in_send_stats_dic = {i: in_send_stats_tot[i] for i in range(len(in_send_stats_tot))}

# out_send_stats = {i:out_send_stats_tot[i] for i in G_dir.nodes}
# in_send_stats = {i:in_send_stats_tot[i] for i in G_dir.nodes} ##wrong node index!!

out_degree = dict(G_dir.out_degree)
in_degree = dict(G_dir.in_degree)
                 
## get the name!                 

users_data = pd.DataFrame.from_dict(first_name, orient='index', columns = ['first_name'])
users_data['tot_msg_sent'] = pd.Series(out_send_stats_dic)
users_data['tot_msg_recieved'] = pd.Series(in_send_stats_dic)
users_data['out_degree'] = pd.Series(out_degree)
users_data['in_degree'] = pd.Series(in_degree)

### get gender
d = gender.Detector()
genderlist = {}
for n in users_data.index:
    first_name = users_data['first_name'][n]
    if first_name == 'Guenther':
        first_name = 'Günther'
    if first_name == 'EvaMaria':
        first_name = 'Eva Maria'
    if first_name == 'Juergen':
        first_name = 'Jürgen'
    gender1 = d.get_gender(first_name)
    genderlist[n] = gender1

users_data['gender'] = pd.Series(genderlist)
dict1 = {}
for n in users_data.index:
    g = users_data['gender'][n]
    # print(g)
    if g in ['female' or 'mostly_female']:
        dict1[n] = 1
        # print(1)
    elif g in ['male' or 'mostly_male']:
        dict1[n] = 0
        # print(0)
users_data['gender_num'] = pd.Series(dict1)


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

users_data.to_csv('users_data.csv')