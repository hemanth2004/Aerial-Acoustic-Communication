_6b_info = """
6-bit encoding used by chirpcomms is a custom character encoding scheme

Contains:
10 numer digits (10)
26 lowercase alphabets (26)
Space, _ (2)
\, !, #, &, @ (5) 
<, > (2)
(, ) (2)
+, -, /, *, = (5)
" (1)
TEXT, FILE, CTRL (3)

- 56 in total
- 8 free

Unspecified numbers will have $ as their dict value.
"""


_6bit_dict = {
    0: '0',  # unused
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'a',
    11: 'b',
    12: 'c',
    13: 'd',
    14: 'e',
    15: 'f',
    16: 'g',
    17: 'h',
    18: 'i',
    19: 'j',
    20: 'k',
    21: 'l',
    22: 'm',
    23: 'n',
    24: 'o',
    25: 'p',
    26: 'q',
    27: 'r',
    28: 's',
    29: 't',
    30: 'u',
    31: 'v',
    32: 'w',
    33: 'x',
    34: 'y',
    35: 'z',
    36: ' ',
    37: '_',
    38: '\\',
    39: '!',
    40: '#',
    41: '&',
    42: '@',
    43: '<',
    44: '>',
    45: '(',
    46: ')',
    47: '+',
    48: '-',
    49: '/',
    50: '*',
    51: '=',
    52: '"',
    53: 'TEXT',
    54: 'FILE',
    55: 'CTRL',
    56: '$',  # unused
    57: '$',  # unused
    58: '$',  # unused
    59: '$',  # unused
    60: '$',  # unused
    61: '$',  # unused
    62: '$',  # unused
    63: '$',  # unused
}


_6b_rev = {v: k for k, v in _6bit_dict.items()}

def char_from_bin(_6bkey):
    return _6bit_dict[_6bkey]
    
def bin_from_char(ch):
    return _6b_rev[ch]

