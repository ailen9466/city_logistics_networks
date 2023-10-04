import numpy as np
import pandas as pd
import pickle


def myfun_com_member_neighbor(city, city_neighbor, com_ix):
    cls_id = np.unique(com_ix)
    com_mb = {}
    com_nb = {}
    for id in cls_id:
        k = com_ix == id
        cls_city = city[k]
        com_mb[id] = cls_city
        nb = []
        for ct in cls_city:
            if len(nb) == 0:
                nb = city_neighbor[ct]
            else:
                nb = np.unique(np.concatenate([nb, city_neighbor[ct]], axis=0))
        com_nb[id] = nb

    return com_mb, com_nb


def myfun_algo_city_cluster(city, city_neighbor, from_city, to_city, weight):
    ## initial community
    num_city = len(city)
    com_ix = np.arange(num_city)
    com_mb, com_nb = myfun_com_member_neighbor(city, city_neighbor, com_ix)

    count = 0
    Q = 0
    Q_record = pd.DataFrame(columns=['Q', 'num_iter'])
    num_iter = 0
    while count <= 500:
        for i in range(num_city):
            DeltaQ = myfun_DeltaQ(city[i], num_city, from_city, to_city, weight, com_ix, com_mb, com_nb)
            maxDeltaQ = DeltaQ.max()
            if maxDeltaQ > 0:
                count = 0
                ix = np.argmax(DeltaQ)
                com_ix[i] = ix
                com_ix = myfun_update_com_ix(com_ix)
                com_mb, com_nb = myfun_com_member_neighbor(city, city_neighbor, com_ix)
                Q = Q + maxDeltaQ
                print(
                    'city {:d} -> community {:d}, DQ = {:.4f}, NComs = {:d}, Q = {:.4f}'.format(city[i], ix, maxDeltaQ,
                                                                                                len(com_mb), Q))
            else:
                count = count + 1
            Q_record = pd.concat([Q_record, pd.DataFrame({'Q': Q, 'num_iter': num_iter}, index=[num_iter])])
            num_iter += 1
    return com_mb, com_ix


def myfun_update_com_ix(com_ix):
    ucom_ix, ix_reverse = np.unique(com_ix, return_inverse=True)
    com_ix = ix_reverse
    return com_ix


def myfun_DeltaQ(move_city, num_city, from_city, to_city, weight, com_ix, com_mb, com_nb):
    num_com = max(com_ix)
    m = np.sum(weight)
    ind_ki = np.isin(from_city, move_city) | np.isin(to_city, move_city)
    k_i = np.sum(weight[ind_ki])

    # isolating a node from a community leads to the change of modularity
    DeltaQ_R = 0
    for i in range(num_com):
        com_member = com_mb[i]
        ind = com_member == move_city
        if len(com_member) > 1 & np.sum(ind) > 0:
            com_member_remove_mcity = com_member[~ind]
            ind_tot = np.isin(from_city, com_member_remove_mcity) | np.isin(to_city, com_member_remove_mcity)
            sigma_tot = np.sum(weight[ind_tot])
            ind_kiin1 = np.isin(from_city, com_member_remove_mcity) & np.isin(to_city, move_city)
            ind_kiin2 = np.isin(to_city, com_member_remove_mcity) & np.isin(from_city, move_city)
            k_iin = np.sum(weight[ind_kiin1]) + np.sum(weight[ind_kiin2])
            DeltaQ_R = (k_iin / m - sigma_tot * k_i / (2 * m ** 2)) * -1
            break

    # sigma_in = np.zeros(num_com)
    sigma_tot = np.zeros(num_com)

    for i in range(num_com):
        com_member = com_mb[i]
        # if com_member.size > 1.5:
        #     ind_in = np.isin(from_city, com_member) & np.isin(to_city, com_member)
        #     sigma_in[i] = np.sum(weight[ind_in])
        if com_member.size < num_city:
            ind_tot = np.isin(from_city, com_member) | np.isin(to_city, com_member)
            sigma_tot[i] = np.sum(weight[ind_tot])

    # adding an isolated node into a community leads to the change of modularity
    DeltaQ = np.zeros(num_com)
    for j in range(num_com):
        com_member = com_mb[j]
        com_neighbor = com_nb[j]
        if (np.sum(np.isin(com_member, move_city)) == 0) & (np.sum(np.isin(com_neighbor, move_city)) > 0):
            ind_kiin1 = np.isin(from_city, com_member) & np.isin(to_city, move_city)
            ind_kiin2 = np.isin(to_city, com_member) & np.isin(from_city, move_city)
            k_iin = np.sum(weight[ind_kiin1]) + np.sum(weight[ind_kiin2])
            DeltaQ[j] = k_iin / m - sigma_tot[j] * k_i / (2 * m ** 2) + DeltaQ_R

    return DeltaQ


if __name__ == '__main__':
    with open('data_city_neighbor.pkl', 'rb') as fid:
        city_neighbor = pickle.load(fid)
        city = pickle.load(fid)
    network_data = pd.read_csv('data_network.csv')
    # city = pd.unique(pd.concat([network_data['from_city'], network_data['to_city']]))
    r1, r2 = myfun_algo_city_cluster(city, city_neighbor, network_data['from_city'], network_data['to_city'],
                                     network_data['weight_etc_cost_city'])
