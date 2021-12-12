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
import pytz
import ast

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


path = '/Users/klaudiamur/Documents/hackathon/hactathon2021'

data_tmp = pd.read_csv(path+'/emails.csv')

with open(path+'/myfile.txt') as f:
    data = f.read()

team_data = ast.literal_eval(data)
chart_tree = np.load(path + '/chart_tree.npy')

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
    
users_list_all = [i for i in senders.values() if 'enron' in i]+[i for i in recipients.values() if 'enron' in i]

users_list, count = np.unique(users_list_all, return_counts=True)
count_sort_ind = np.argsort(-count)
users_list = users_list[count_sort_ind]
email = {i:n for i, n in enumerate(users_list)}
first_name = {i:n.split('.')[0].capitalize() for i, n in enumerate(users_list)}

users_data_tmp = pd.DataFrame.from_dict(first_name, orient='index', columns = ['first_name'])
users_data_tmp['Email'] = pd.Series(email)

### get gender based on first name
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

    if g in ['female', 'mostly_female']:
        dict1[n] = 1

    elif g in ['male', 'mostly_male']:
        dict1[n] = 0

users_data_tmp['gender_num'] = pd.Series(dict1)

### pick only the ones that have a first name!
d = users_data_tmp['gender_num'].to_dict()
human_users_indx = [i for i, n in genderlist.items() if n != 'unknown'] ### check so that mostly_female becomes 1
human_users = [email[i] for i in human_users_indx]   
users_data =  pd.DataFrame.from_dict({i:n for i, n in enumerate(human_users)}, orient='index', columns= ['Email'])
users_data['first_name'] = pd.Series({i:users_data_tmp['first_name'][n] for i, n in enumerate(human_users_indx)})
users_data['gender'] = pd.Series({i:users_data_tmp['gender'][n] for i, n in enumerate(human_users_indx)})
users_data['gender_num'] = pd.Series({i:users_data_tmp['gender_num'][n] for i, n in enumerate(human_users_indx)})

### pick only 500 to make the network smaller
size = 500    
dir_matrix_listed = make_time_network(human_users[:size], data, human_users[:size])  

nw_tot = np.sum(dir_matrix_listed['nw'], axis=2)

#Gdir is the directed network that includes all (weighted) email exchanges between two users
G_dir = nx.from_numpy_matrix(nw_tot, create_using=nx.DiGraph)
emails_to_themselves_or_no_emails = list(nx.isolates(G_dir))
G_dir.remove_nodes_from(list(nx.isolates(G_dir)))

## G1: two people have a connection if they have exchanged at least 5 emails (undirected network)
G1, G1_matrix = make_tight_nw(dir_matrix_listed, 7, 1, 5)
## G1: two people have a connection if they have exchanged emails in at least 12 14-day periods (undirected network)
G2, G2_matrix = make_tight_nw(dir_matrix_listed, 14, 12, 2)
  
## calculte statistics
out_send_stats_tot = np.sum(nw_tot, axis=1)
in_send_stats_tot = np.sum(nw_tot, axis=0)
out_send_stats_dic = {i: out_send_stats_tot[i] for i in range(len(out_send_stats_tot))}
in_send_stats_dic = {i: in_send_stats_tot[i] for i in range(len(in_send_stats_tot))}


out_degree = dict(G_dir.out_degree)
in_degree = dict(G_dir.in_degree)
                 
## collect them in the dataframe               
users_data['tot_msg_sent'] = pd.Series(out_send_stats_dic)
users_data['tot_msg_recieved'] = pd.Series(in_send_stats_dic)
users_data['out_degree'] = pd.Series(out_degree)
users_data['in_degree'] = pd.Series(in_degree)


### run network measurements on the two undirected networks, get datapoint for every user
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

### calulate number and rate of emails sent on weekends and after working hours
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

            t = row['ts_or'] ## this is the local time (of sender)

            day = t.date().weekday()
            hour = t.hour

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

users_data['overwork_tot_1'] = over_work['weekends'] + over_work['after_wh']

users_data.loc[users_data['tot_msg_sent'] < 10, 'overwork_ratio_1'] = 0

### calculate effective length of days by taking time difference between 
### first email of the day and last email, counting all days that are longer than 8.5 hours
ts_list = {}
ts_len = {}
df_dic = data.to_dict('records')
for row in df_dic: #use their local time?
    sen = row['sender']    
    if sen in human_users:
        indx0 = users_data[users_data['Email'] == sen].index.values
        if len(indx0) > 0:
            indx = indx0[0]
            t = row['ts_or']
            if not indx in ts_list.keys():
                ts_list[indx] = []
            ts_list[indx] = ts_list[indx] + [t]
            
for user, tsl in ts_list.items():
    tsl.sort()
    l_of_day = []
    day = tsl[0].date()
    h0 = tsl[0]
    for t in tsl:
        if t.date() == day:
            hmax = t
        else:
            day = t.date()
            l_day = hmax.astimezone(pytz.UTC) - h0.astimezone(pytz.UTC)
            l_of_day.append(l_day)
            h0 = t
            hmax = t
    ts_len[user] = l_of_day
    
## count days that are longer than 8,5 hours
t0 = datetime.timedelta(hours=8, minutes=30)
len_days_tot = {n:(len([t for t in tl if t > t0])) for n, tl in ts_len.items()}           
users_data['n_long_days'] = pd.Series(len_days_tot)  

# =============================================================================
# analyse formal hierarchy and teams
# =============================================================================
users_data['teamname'] = pd.Series(team_data)
boss_indx = 0
for i in range(len(chart_tree)):    
    if np.sum(chart_tree, axis = 0)[i] == 0:
        if np.sum(chart_tree, axis = 1)[i] > 0:
            boss_indx = i
            
          
formal_hierarchy_tree = nx.from_numpy_matrix(chart_tree, create_using=nx.DiGraph())
formal_hierarchy_tree.remove_nodes_from(list(nx.isolates(formal_hierarchy_tree)))
### find the ones that are in the tree but not in the email list:
indx = [n for n in range(len(users_data)) if n not in formal_hierarchy_tree.nodes()]    
not_in_chart = users_data.iloc[indx]

people_below = {}
n_direct_managing = {}
depth_below = {}
heigth = {}
manager= {}
for n in formal_hierarchy_tree.nodes():
    sp = nx.shortest_path(formal_hierarchy_tree, source = boss_indx, target = n)
    if len(sp) > 2:
        manager[n] = sp[-2]
    heigth[n] = len(sp)
    #heigth[n] = nx.shortest_path_length(formal_hierarchy_tree, source = boss_indx, target = n)
    reachable_nodes = nx.single_source_shortest_path_length(formal_hierarchy_tree, n)
    people_below[n] = len(reachable_nodes) -1 
    n_direct_managing[n] = len( [k for k, v in reachable_nodes.items() if v == 1])
    depth_below[n] = max(reachable_nodes.values())
    
users_data['people_below']=pd.Series(people_below)   
users_data['n_direct_managing'] = pd.Series(n_direct_managing)
users_data['depth_below'] = pd.Series(depth_below)
users_data['heigth'] = pd.Series(heigth)
users_data['manager'] = pd.Series(manager)



# =============================================================================
# more measurement with network + hierarchy data
# =============================================================================
for name, ntwork in networks.items():
    div_teams = {}
    #ratio own team - other teams
    ratio_div_t = {}
    div_hier = {} ## how many of the connections are to higher up, how many down
    
    
    for i in ntwork.nodes:
        nb = [k for k in  ntwork.neighbors(i)]
        team_nb = [users_data['teamname'][n] for n in nb if not pd.isna(users_data['teamname'][n])] ## only if not nan!
        h_nb = [users_data['heigth'][n] for n in nb if not pd.isna(users_data['heigth'][n])]
        div_hier[i] = np.mean(h_nb) - users_data['heigth'][i]

        
        div_t_tot = len(team_nb)
        if div_t_tot > 0:
            div_t = [l for l in team_nb if l != users_data['teamname'][i]]        
            ratio_div_t[i] = 1 - len(div_t)/div_t_tot
            div_tea = len(np.unique(div_t))
            div_teams[i] = div_tea
            

    users_data['div_interaction_teams_' + name] = pd.Series(div_teams)
    users_data['div_hier_'+name] = pd.Series(div_hier)
    users_data['ratio_div_t_'+name] = pd.Series(ratio_div_t)
    
    # if network is connected:
    #if name == 'G1':
    #    dis_0 = {}
    #    for i in ntwork.nodes():
    #        sp = nx.shortest_path_length(ntwork, source = boss_indx, target = i)
    #        dis_0[i] = sp
    #    users_data['dis_from_CEO_G1'] = pd.Series(dis_0)        
    
 
    results_dict = {}
    for n in ntwork.nodes():
        gender_ratio = np.mean([users_data['gender_num'][n1] for n1 in ntwork.neighbors(n) if not pd.isna(users_data['gender_num'][n1]) ])
        results_dict[n] = gender_ratio
        
    users_data['homophiliy_'+name] = pd.Series(results_dict)  

## save users_data to further analyse
users_data.to_csv(path + '/users_data.csv')

