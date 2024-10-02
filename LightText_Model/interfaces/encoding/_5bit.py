_5b_info = """
5-bit encoding used by chirpcomms is a custom character encoding scheme

Contains:
26 lowercase alphabets (26)
space, \\ (2) (special)
[, ], <, > (4) (delimiters)
 
32/32
"""


_5bit_dict = {
    0: 'a', 
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h',
    8: 'i',
    9: 'j',
    10: 'k',
    11: 'l',
    12: 'm',
    13: 'n',
    14: 'o',
    15: 'p',
    16: 'q',
    17: 'r',
    18: 's',
    19: 't',
    20: 'u',
    21: '[', # 21 in binary is 10101 which helps with sync as a preamble
    22: ']', # 10110
    23: '<', # another optional preamble
    24: 'y',
    25: 'z',
    26: ' ',
    27: '\\',
    28: 'v',
    29: 'w',
    30: 'x',
    31: '>',
}

_5b_rev = {v: k for k, v in _5bit_dict.items()}
_5b_charlist = _5b_rev.keys()

def char_from_bin(_6bkey):
    return _5bit_dict[_6bkey]
    
def bin_from_char(ch):
    return _5b_rev[ch]

def _5b_verify(txt):
    for ch in txt:
        if ch not in _5b_charlist:
            return False

    return True

