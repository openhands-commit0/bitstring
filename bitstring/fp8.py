"""
The 8-bit float formats used here are from a proposal supported by Graphcore, AMD and Qualcomm.
See https://arxiv.org/abs/2206.02915

"""
import struct
import zlib
import array
import bitarray
from bitstring.luts import binary8_luts_compressed
import math

class Binary8Format:
    """8-bit floating point formats based on draft IEEE binary8"""

    def __init__(self, exp_bits: int, bias: int):
        self.exp_bits = exp_bits
        self.bias = bias
        self.pos_clamp_value = 127
        self.neg_clamp_value = 255

    def __str__(self):
        return f'Binary8Format(exp_bits={self.exp_bits}, bias={self.bias})'

    def float_to_int8(self, f: float) -> int:
        """Given a Python float convert to the best float8 (expressed as an integer in 0-255 range)."""
        if math.isnan(f):
            return 0
        if math.isinf(f):
            if f > 0:
                return self.pos_clamp_value
            return self.neg_clamp_value
        if f == 0:
            return 0
        sign = 1 if f < 0 else 0
        f = abs(f)
        exp = math.floor(math.log2(f))
        mantissa = int((f / 2**exp - 1) * (1 << (7 - self.exp_bits)))
        exp = exp + self.bias
        if exp < 0:
            return 0
        if exp >= (1 << self.exp_bits):
            return self.neg_clamp_value if sign else self.pos_clamp_value
        result = (sign << 7) | (exp << (7 - self.exp_bits)) | mantissa
        return result

    def createLUT_for_binary8_to_float(self) -> array.array:
        """Create a LUT to convert an int in range 0-255 representing a float8 into a Python float"""
        lut = array.array('f')
        for i in range(256):
            sign = -1 if i & 0x80 else 1
            exp = (i >> (7 - self.exp_bits)) & ((1 << self.exp_bits) - 1)
            mantissa = i & ((1 << (7 - self.exp_bits)) - 1)
            if exp == 0:
                lut.append(0.0)
            else:
                value = sign * (1 + mantissa / (1 << (7 - self.exp_bits))) * 2**(exp - self.bias)
                lut.append(value)
        return lut
p4binary_fmt = Binary8Format(exp_bits=4, bias=8)
p3binary_fmt = Binary8Format(exp_bits=5, bias=16)

def decompress_luts() -> None:
    """Decompress the lookup tables for binary8 formats."""
    for fmt in [p4binary_fmt, p3binary_fmt]:
        if not hasattr(fmt, 'lut_float16_to_binary8'):
            key = (fmt.exp_bits, fmt.bias)
            compressed_data = binary8_luts_compressed[key]
            fmt.lut_float16_to_binary8 = zlib.decompress(compressed_data[0])
        if not hasattr(fmt, 'lut_binary8_to_float'):
            fmt.lut_binary8_to_float = fmt.createLUT_for_binary8_to_float()