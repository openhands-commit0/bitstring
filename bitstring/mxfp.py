import array
import math
import struct
import bitarray
from bitstring.luts import mxfp_luts_compressed
import zlib
from typing import Optional

class MXFPFormat:
    """Defining an MXFP micro-scaling floating point format"""

    def __init__(self, exp_bits: int, mantissa_bits: int, bias: int, mxfp_overflow: str):
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias
        self.mxfp_overflow = mxfp_overflow
        self.pos_clamp_value = (1 << self.exp_bits + self.mantissa_bits) - 1
        self.neg_clamp_value = (1 << 1 + self.exp_bits + self.mantissa_bits) - 1
        if self.exp_bits == 4 and self.mantissa_bits == 3:
            if self.mxfp_overflow == 'saturate':
                self.pos_clamp_value = 126
                self.neg_clamp_value = 254
            else:
                self.pos_clamp_value = self.neg_clamp_value = 255
        if self.exp_bits == 5 and self.mantissa_bits == 2:
            if self.mxfp_overflow == 'saturate':
                self.pos_clamp_value = 123
                self.neg_clamp_value = 251
            else:
                self.pos_clamp_value = 124
                self.neg_clamp_value = 252
        self.lut_float16_to_mxfp = None
        self.lut_int_to_float = None

    def __str__(self):
        return f"MXFPFormat(exp_bits={self.exp_bits}, mantissa_bits={self.mantissa_bits}, bias={self.bias}, mxfp_overflow='{self.mxfp_overflow}')"

    def float_to_int(self, f: float) -> int:
        """Given a Python float convert to the best mxfp float (expressed as an int) that represents it."""
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
        mantissa = int((f / 2**exp - 1) * (1 << self.mantissa_bits))
        exp = exp + self.bias
        if exp < 0:
            return 0
        if exp >= (1 << self.exp_bits):
            if self.mxfp_overflow == 'saturate':
                return self.neg_clamp_value if sign else self.pos_clamp_value
            exp = (1 << self.exp_bits) - 1
            mantissa = (1 << self.mantissa_bits) - 1
        result = (sign << (self.exp_bits + self.mantissa_bits)) | (exp << self.mantissa_bits) | mantissa
        return result

    def createLUT_for_int_to_float(self) -> array.array:
        """Create a LUT to convert an int in representing a MXFP float into a Python float"""
        lut = array.array('f')
        for i in range(1 << (1 + self.exp_bits + self.mantissa_bits)):
            sign = -1 if i >> (self.exp_bits + self.mantissa_bits) else 1
            exp = (i >> self.mantissa_bits) & ((1 << self.exp_bits) - 1)
            mantissa = i & ((1 << self.mantissa_bits) - 1)
            if exp == 0:
                lut.append(0.0)
            else:
                value = sign * (1 + mantissa / (1 << self.mantissa_bits)) * 2**(exp - self.bias)
                lut.append(value)
        return lut

    def createLUT_for_float16_to_mxfp(self) -> bytes:
        """Create a LUT to convert a float16 into a MXFP format"""
        lut = bytearray(65536)
        for i in range(65536):
            f = struct.unpack('e', struct.pack('H', i))[0]
            lut[i] = self.float_to_int(f)
        return bytes(lut)
e2m1mxfp_fmt = MXFPFormat(exp_bits=2, mantissa_bits=1, bias=1, mxfp_overflow='saturate')
e2m3mxfp_fmt = MXFPFormat(exp_bits=2, mantissa_bits=3, bias=1, mxfp_overflow='saturate')
e3m2mxfp_fmt = MXFPFormat(exp_bits=3, mantissa_bits=2, bias=3, mxfp_overflow='saturate')
e4m3mxfp_saturate_fmt = MXFPFormat(exp_bits=4, mantissa_bits=3, bias=7, mxfp_overflow='saturate')
e5m2mxfp_saturate_fmt = MXFPFormat(exp_bits=5, mantissa_bits=2, bias=15, mxfp_overflow='saturate')
e4m3mxfp_overflow_fmt = MXFPFormat(exp_bits=4, mantissa_bits=3, bias=7, mxfp_overflow='overflow')
e5m2mxfp_overflow_fmt = MXFPFormat(exp_bits=5, mantissa_bits=2, bias=15, mxfp_overflow='overflow')

def decompress_luts() -> None:
    """Decompress the lookup tables for MXFP formats."""
    for fmt in [e2m1mxfp_fmt, e2m3mxfp_fmt, e3m2mxfp_fmt, e4m3mxfp_saturate_fmt, e5m2mxfp_saturate_fmt, e4m3mxfp_overflow_fmt, e5m2mxfp_overflow_fmt]:
        if fmt.lut_float16_to_mxfp is None:
            key = (fmt.exp_bits, fmt.mantissa_bits, fmt.bias, fmt.mxfp_overflow)
            fmt.lut_float16_to_mxfp = zlib.decompress(mxfp_luts_compressed[key])
        if fmt.lut_int_to_float is None:
            fmt.lut_int_to_float = fmt.createLUT_for_int_to_float()