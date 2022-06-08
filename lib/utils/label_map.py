class ScannetLabel:
    label_map_Nyu40 = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
        11: 10, 12: 11, 13: 255, 14: 12, 15: 255, 16: 13, 17: 255, 18: 255, 19: 255,
        20: 255, 21: 255, 22: 255, 23: 255, 24: 14, 25: 255, 26: 255, 27: 255, 28: 15, 29: 255,
        30: 255, 31: 255, 32: 255, 33: 16, 34: 17, 35: 255, 36: 18, 37: 255, 38: 255, 39: 19,
        40: 255
    }

def convert_label(label, map, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in map.items():
            label[temp == k] = v
    else:
        for k, v in map.items():
            label[temp == k] = v
    return label
