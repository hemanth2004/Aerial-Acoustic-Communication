import numpy as np
from sklearn.metrics import accuracy_score

from signals.util import num2binarr as d2b
from encoding._5bit import char_from_bin

def get_frame(received, original):
    print("Received: ",  received)
    print("Original: ", original)
    acc = accuracy_score(original, received)
    print("--------")
    print("Bit Accuracy: ", (acc*100), "%")

    received = received[1:]
    original = original[1:]

    msg = ""
    for i in range(0, len(received), 5):
        binary_chunk = received[i:i + 5]
        num = d2b.fro(binary_chunk)
        char = char_from_bin(num)
        msg += char
    
    orig_msg = ""
    for i in range(0, len(original), 5):
        binary_chunk = original[i:i + 5]
        num = d2b.fro(binary_chunk)
        char = char_from_bin(num)
        orig_msg += char

    print("Received Msg: ", msg)
    # print("Original Msg: ", orig_msg)
    return msg