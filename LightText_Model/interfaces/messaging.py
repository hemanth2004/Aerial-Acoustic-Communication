import numpy as np
from signals.util import num2binarr
from encoding._5bit import char_from_bin, bin_from_char, _5b_verify
from signals.errors.checksum import append_checksum

from signals.sender import send_bits
from signals.modulation.ask import generate_ask_signal, decode_ask_signal
from signals.modulation.css import generate_css_bok_signal_half_range, generate_css_bok_signal, generate_css_cts_signal


message_txt = "hi"
symbol_duration = 0.13
symbol_gap = 0.03


send_encoding = bin_from_char
receive_encoding = char_from_bin

mod_fn = generate_css_bok_signal
mod_order = 2

bit_per_symbol = 5
verify_fn = _5b_verify

ask_preamble = [1, 1, 1, 1, 0]
preamble_bits = [1, 1, 0, 1]

preamble_signal = generate_css_bok_signal(
    bit_array=preamble_bits,
    sample_rate=44100, symbol_duration=symbol_duration, gap=symbol_gap)

def send_text(txt):
    if not verify_fn(txt):
        return
    
    enc_arr = [num2binarr.to(send_encoding(char), bit_per_symbol) for char in txt]
    # For addding error checking checksum
    # Replace with error correction method isntead
    # enc_arr = append_checksum(enc_arr, bit_per_symbol)
    enc_arr_np = np.array(enc_arr).flatten()

    # if mod_fn == generate_css_bok_signal: 
        # if using CSS modulation
        # then re transmit last symbol because the last bit never registers 
        # on the receiver's correlator for whatever reason
        #enc_arr_np = np.append(enc_arr_np, [enc_arr_np[-1]]) 
        

    print("sent data: ", enc_arr)
    print("bits sent: ", len(enc_arr_np))

    f = open("__comm.txt", "w")
    if mod_fn == generate_ask_signal:
        f.write(str(0))
    for i in enc_arr_np:
        f.write(str(i))
    f.close()

    send_bits(enc_arr_np, mod_fn, symbol_duration, plot_wave=False, preamble_signal=preamble_signal, gap=symbol_gap)


if __name__ == "__main__":
    send_text(message_txt)
