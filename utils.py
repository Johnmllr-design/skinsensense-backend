# a function for parsing values of an array into buckets
def to_hash(arr):
    g_hash = {}
    i = 0
    p = 0
    placed = False
    for value in arr:
        if value <= 0.1:
            if 0.1 not in g_hash.keys():
                placed = True
                g_hash[0.1] = 1
            else:
                placed = True
                g_hash[0.1] += 1
        elif value > 0.1 and value <= 0.2:
            if 0.2 not in g_hash.keys():
                placed = True
                g_hash[0.2] = 1
            else:
                g_hash[0.2] += 1
                placed = True
        elif value > 0.2 and value <= 0.3:
            if 0.3 not in g_hash.keys():
                placed = True
                g_hash[0.3] = 1
            else:
                g_hash[0.3] += 1
                placed = True
        elif value > 0.3 and value <= 0.4:
            if 0.4 not in g_hash.keys():
                g_hash[0.4] = 1
                placed = True
            else:
                g_hash[0.4] += 1
                placed = True
        elif value > 0.4 and value <= 0.5:
            if 0.5 not in g_hash.keys():
                placed = True
                g_hash[0.5] = 1
            else:
                g_hash[0.5] += 1
                placed = True
        elif value > 0.5 and value <= 0.6:
            if 0.6 not in g_hash.keys():
                placed = True
                g_hash[0.6] = 1
            else:
                placed = True
                g_hash[0.6] += 1
        elif value > 0.6 and value <= 0.7:
            if 0.7 not in g_hash.keys():
                placed = True
                g_hash[0.7] = 1
            else:
                g_hash[0.7] += 1
                placed = True
        elif value > 0.7 and value <= 0.8:
            if 0.8 not in g_hash.keys():
                placed = True
                g_hash[0.8] = 1
            else:
                placed = True
                g_hash[0.8] += 1
        elif value > 0.8 and value <= 0.9:
            if 0.9 not in g_hash.keys():
                placed = True
                g_hash[0.9] = 1
            else:
                placed = True
                g_hash[0.9] += 1
        elif value > 0.9 and value <= 1:
            if 1 not in g_hash.keys():
                placed = True
                g_hash[1] = 1
            else:
                placed = True
                g_hash[1] += 1
        i += 1
        if placed:
            p += 1
    return g_hash




