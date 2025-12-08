def add_padding(list,max_length):
    padd_num = max_length - len(list)
    padded_list = list + [0]*padd_num
    return padded_list