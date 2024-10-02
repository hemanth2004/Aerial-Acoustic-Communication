import numpy as np
from signals.util import num2binarr

from encoding._5bit import char_from_bin, bin_from_char, _5b_verify
from signals.errors.checksum import append_checksum

from signals.sender import send_bits
from signals.receiver_local import recorded_read
from signals.modulation.ask import generate_ask_signal, decode_ask_signal


message_txt = "there"


send_encoding = bin_from_char
receive_encoding = char_from_bin

mod_fn = generate_ask_signal
demod_fn = decode_ask_signal

mod_order = 2

bit_per_symbol = 5
verify_fn = _5b_verify

preamble_bits = [1, 1, 1, 1, 0]

preamble_signal = generate_ask_signal(
    bit_array=preamble_bits,
    sample_rate=44100, symbol_duration=0.2)

def send_text(txt):
    if not verify_fn(txt):
        return
    
    enc_arr = [num2binarr.to(send_encoding(char), bit_per_symbol) for char in txt]
    # enc_arr = append_checksum(enc_arr, bit_per_symbol)
    enc_arr_np = np.array(enc_arr).flatten()

    print("sent data: ", enc_arr)
    print("bits sent: ", len(enc_arr_np))

    f = open("__comm.txt", "w")
    f.write(str(0))
    for i in enc_arr_np:
        f.write(str(i))
    f.close()

    send_bits(enc_arr_np, mod_fn, plot_wave=False, preamble_signal=preamble_signal)

def receive_text(bits):
    print(bits)

def decode_recorded():

    bin_arr = recorded_read(demod_fn, "ask_signal")
    bin_arr = np.array(bin_arr).reshape(-1, bit_per_symbol)

    msg = [receive_encoding(num2binarr.fro(bin)) for bin in bin_arr]
    msg = ''.join(msg)
    print(msg)
    
if __name__ == "__main__":
    send_text(message_txt)
