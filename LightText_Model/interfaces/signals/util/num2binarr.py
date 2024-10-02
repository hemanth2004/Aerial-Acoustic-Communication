def to(num, fit = -1):
    bin_str = bin(num)[2:]

    bin_arr = [int(char) for char in bin_str]

    if fit != -1:
        if len(bin_arr) <= fit:
            cnt = fit - len(bin_arr)
            for _ in range(cnt):
                bin_arr.insert(0, 0)
        else:
            print("(wng: num2binarr fit is lower than digits)")

    return bin_arr

def fro(bin_arr):
    num = 0
    for bit in bin_arr:
        num = (num << 1) | bit
    return num