import numpy as np

def get_idx_list(action, dealer, player):
    idx_list = []
    idx = 0
    if action == 0 and dealer in range(1,5) and player in range(1,7):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(4,8) and player in range(1,7):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(7,11) and player in range(1,7):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(1,5) and player in range(4,10):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(4,8) and player in range(4,10):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(7,11) and player in range(4,10):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(1,5) and player in range(7,13):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(4,8) and player in range(7,13):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(7,11) and player in range(7,13):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(1,5) and player in range(10,16):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(4,8) and player in range(10,16):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(7,11) and player in range(10,16):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(1,5) and player in range(13,19):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(4,8) and player in range(13,19):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(7,11) and player in range(13,19):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(1,5) and player in range(16,22):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(4,8) and player in range(16,22):
        idx_list.append(idx)
    idx += 1
    if action == 0 and dealer in range(7,11) and player in range(16,22):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(1,5) and player in range(1,7):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(4,8) and player in range(1,7):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(7,11) and player in range(1,7):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(1,5) and player in range(4,10):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(4,8) and player in range(4,10):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(7,11) and player in range(4,10):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(1,5) and player in range(7,13):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(4,8) and player in range(7,13):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(7,11) and player in range(7,13):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(1,5) and player in range(10,16):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(4,8) and player in range(10,16):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(7,11) and player in range(10,16):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(1,5) and player in range(13,19):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(4,8) and player in range(13,19):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(7,11) and player in range(13,19):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(1,5) and player in range(16,22):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(4,8) and player in range(16,22):
        idx_list.append(idx)
    idx += 1
    if action == 1 and dealer in range(7,11) and player in range(16,22):
        idx_list.append(idx)
    idx += 1
    return idx_list



def get_feature():
    # setup approx. feature lookup table
    feature = {}
    for action in [0, 1]:
        for dealer in range(1, 11):
            for player in range(1, 22):
                feature[(action, dealer, player)] = np.zeros(36, dtype=int)
                idx_list = get_idx_list(action, dealer, player)
                for idx in idx_list:
                    feature[(action, dealer, player)][idx] = 1
    return feature