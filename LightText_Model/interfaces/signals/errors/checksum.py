import numpy as np

def flatten_bit_array(bit_array_2d):
    return [bit for sublist in bit_array_2d for bit in sublist]

def get_checksum(bit_array_2d, div, return_bit_array=False):
    bit_array = flatten_bit_array(bit_array_2d)
    n = len(bit_array)

    if n % div != 0:
        raise ValueError("Divisor must be a factor of the length of the bit array")

    chunks = [bit_array[i:i + div] for i in range(0, n, div)]
    checksum = [0] * div
    for chunk in chunks:
        for i in range(div):
            checksum[i] ^= chunk[i]

    if return_bit_array:
        return checksum
    else:
        return int(''.join(map(str, checksum)), 2)

def append_checksum(bit_array_2d, div):
    bit_array = flatten_bit_array(bit_array_2d)
    checksum = get_checksum(bit_array_2d, div, return_bit_array=True)
    new_bit_array = bit_array + checksum

    num_columns = len(bit_array_2d[0])
    num_rows = len(new_bit_array) // num_columns
    new_bit_array_2d = np.reshape(new_bit_array, (num_rows, num_columns)).tolist()

    return new_bit_array_2d

def verify_checksum(bit_array, div, checksum):
    bit_array_len = len(bit_array)
    if bit_array_len % div != 0:
        raise ValueError("Divisor must be a factor of the length of the bit array")

    chunks = [bit_array[i:i + div] for i in range(0, bit_array_len, div)]
    calculated_checksum = [0] * div
    for chunk in chunks:
        for i in range(div):
            calculated_checksum[i] ^= chunk[i]

    return calculated_checksum == checksum


def verify_checksum_2d(bit_array_2d, checksum):
    div = len(bit_array_2d[0])
    bit_array = flatten_bit_array(bit_array_2d)
    return verify_checksum(bit_array, div, checksum)
