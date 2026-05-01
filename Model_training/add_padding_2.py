def add_padding_2(list,max_length):
    padded_list = [None, None]
    padd_num = max_length - len(list[0])
    padded_list[0] = list[0] + [0]*padd_num
    padded_list[1] = list[1] + [0]*padd_num
    return padded_list