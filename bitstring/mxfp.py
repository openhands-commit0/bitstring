import array
import math
import struct
import bitarray
from bitstring.luts import mxfp_luts_compressed
import zlib


def createLUT_for_int_to_float(exp_bits: int, mantissa_bits: int, bias: int) -> array.array:
    """Create a LUT to convert an int in representing a MXFP float into a Python float"""
    i2f = []
    length = 1 + exp_bits + mantissa_bits
    for i in range(1 << length):
        b = bitarray.util.int2ba(i, length=length, endian='big', signed=False)
        sign = b[0]
        exponent = bitarray.util.ba2int(b[1:1 + exp_bits])
        significand = b[1 + exp_bits:]
        if exponent == 0:
            significand = bitarray.bitarray('0') + significand
            exponent = -bias + 1
        else:
            significand = bitarray.bitarray('1') + significand
            exponent -= bias
        f = float(bitarray.util.ba2int(significand)) / (2.0 ** mantissa_bits)
        f *= 2 ** exponent
        if length == 8:
            # Some special cases
            if exp_bits == 5:
                if i in [0b01111100, 0b11111100]:
                    f = float('inf')
                if i in [0b01111101, 0b11111101, 0b01111110, 0b11111110, 0b01111111, 0b11111111]:
                    f = float('nan')
            if exp_bits == 4:
                if i in [0b01111111, 0b11111111]:
                    f = float('nan')
        i2f.append(f if not sign else -f)
    return array.array('f', i2f)


def createLUT_for_float16_to_mxfp(lut_int_to_float, exp_bits: int, mantissa_bits: int, bias: int) -> bytes:
    """Create a LUT to convert a float16 into a MXFP format"""
    # Used to create the LUT that was compressed and stored for the fp8 code
    fp16_to_fp8 = bytearray(1 << 16)
    for i in range(1 << 16):
        b = struct.pack('>H', i)
        f, = struct.unpack('>e', b)
        fp8_i = slow_float_to_int(f, lut_int_to_float, exp_bits, mantissa_bits, bias)
        if fp8_i == 1 << (exp_bits + mantissa_bits):
            # Got back int representing binary digits for negative zero. Just convert to positive zero instead.
            fp8_i = 0
        fp16_to_fp8[i] = fp8_i
    return bytes(fp16_to_fp8)

def slow_float_to_int(f: float, lut_int_to_float, exp_bits: int, mantissa_bits: int, bias: int) -> int:
    # Slow, but easier to follow than the faster version.
    # The output int has the binary sequence needed for the float.
    length = 1 + exp_bits + mantissa_bits
    values = 1 << length
    if f >= 0:
        # Positive, so top bit is not set
        for i in range(values // 2 - 1):
            lower = lut_int_to_float[i]
            upper = lut_int_to_float[i + 1]
            if f == lower:
                return i
            if f == upper:
                return i + 1
            if lower < f < upper:
                d1 = f - lower
                d2 = upper - f
                if d1 < d2:
                    return i
                if d2 < d1:
                    return i + 1
                return i if i % 2 == 0 else i + 1
        # Clip to positive max
        return (1 << (length - 1)) - 1
    if f < 0:
        # Negative, so top bit is set
        for i in range(values // 2, values - 1):
            upper = lut_int_to_float[i]
            lower = lut_int_to_float[i + 1]
            if f == lower:
                return i + 1
            if f == upper:
                return i
            if lower < f < upper:
                d1 = f - lower
                d2 = upper - f
                if d1 < d2:
                    return i + 1
                if d2 < d1:
                    return i
                return i if i % 2 == 0 else i + 1
        # Clip to negative max
        return (1 << length) - 1
    if math.isnan(f):
        return 0  # Nan isn't supported so what value should this be? (TODO)


class MXFPFormat:
    """Defining an MXFP micro-scaling floating point format"""

    def __init__(self, exp_bits: int, mantissa_bits: int, bias: int):
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias

        int_to_float_compressed, float16_to_mxfp_compressed = mxfp_luts_compressed[(exp_bits, mantissa_bits, bias)]
        self.lut_float16_to_mxfp = zlib.decompress(float16_to_mxfp_compressed)
        dec = zlib.decompress(int_to_float_compressed)
        self.lut_int_to_float = struct.unpack(f'<{len(dec) // 4}f', dec)

    def float_to_int(self, f: float) -> int:
        """Given a Python float convert to the best mxfp float (expressed as an int) that represents it."""
        # First convert the float to a float16, then a 16 bit uint
        try:
            b = struct.pack('>e', f)
        except (OverflowError, struct.error):
            # Return the largest representable positive or negative value
            # Special cases for e4m3 and e5m2
            if self.exp_bits == 4 and self.mantissa_bits == 3:
                return 0b01111110 if f > 0 else 0b11111110
            if self.exp_bits == 5 and self.mantissa_bits == 2:
                return 0b01111011 if f > 0 else 0b11111011
            return (1 << (self.exp_bits + self.mantissa_bits)) - 1 if f > 0 else (1 << (1 + self.exp_bits + self.mantissa_bits)) - 1
        f16_int = int.from_bytes(b, byteorder='big')
        # Then use this as an index to our large LUT
        return self.lut_float16_to_mxfp[f16_int]


e2m1mxfp_fmt = MXFPFormat(exp_bits=2, mantissa_bits=1, bias=1)
e2m3mxfp_fmt = MXFPFormat(exp_bits=2, mantissa_bits=3, bias=1)
e3m2mxfp_fmt = MXFPFormat(exp_bits=3, mantissa_bits=2, bias=3)
e4m3mxfp_fmt = MXFPFormat(exp_bits=4, mantissa_bits=3, bias=7)
e5m2mxfp_fmt = MXFPFormat(exp_bits=5, mantissa_bits=2, bias=15)