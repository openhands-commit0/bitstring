from __future__ import annotations
import struct
import math
import functools
from typing import Union, Optional, Dict, Callable
import bitarray
from bitstring.bitstore import BitStore
import bitstring
from bitstring.fp8 import p4binary_fmt, p3binary_fmt
from bitstring.mxfp import e3m2mxfp_fmt, e2m3mxfp_fmt, e2m1mxfp_fmt, e4m3mxfp_saturate_fmt, e5m2mxfp_saturate_fmt, e4m3mxfp_overflow_fmt, e5m2mxfp_overflow_fmt
CACHE_SIZE = 256

def tidy_input_string(s: str) -> str:
    """Return string made lowercase and with all whitespace and underscores removed."""
    return s.lower().replace('_', '').replace(' ', '')
e8m0mxfp_allowed_values = [float(2 ** x) for x in range(-127, 128)]

def hex2bitstore(s: str) -> BitStore:
    """Convert a hex string to a BitStore."""
    s = tidy_input_string(s)
    if not s.startswith('0x'):
        raise ValueError("Hex string must start with '0x'")
    s = s[2:]  # Remove '0x'
    # Each hex digit represents 4 bits
    bits = ''.join(format(int(c, 16), '04b') for c in s)
    ba = bitarray.bitarray(bits)
    return BitStore(ba)

def bin2bitstore(s: str) -> BitStore:
    """Convert a binary string to a BitStore."""
    s = tidy_input_string(s)
    if not s.startswith('0b'):
        raise ValueError("Binary string must start with '0b'")
    s = s[2:]  # Remove '0b'
    ba = bitarray.bitarray(s)
    return BitStore(ba)

def oct2bitstore(s: str) -> BitStore:
    """Convert an octal string to a BitStore."""
    s = tidy_input_string(s)
    if not s.startswith('0o'):
        raise ValueError("Octal string must start with '0o'")
    s = s[2:]  # Remove '0o'
    # Each octal digit represents 3 bits
    bits = ''.join(format(int(c, 8), '03b') for c in s)
    ba = bitarray.bitarray(bits)
    return BitStore(ba)

literal_bit_funcs: Dict[str, Callable[..., BitStore]] = {'0x': hex2bitstore, '0X': hex2bitstore, '0b': bin2bitstore, '0B': bin2bitstore, '0o': oct2bitstore, '0O': oct2bitstore}

def bitstore_from_token(token: str) -> BitStore:
    """Create a BitStore from a token string.
    
    The token can be a hex, binary or octal string.
    """
    token = tidy_input_string(token)
    for prefix, func in literal_bit_funcs.items():
        if token.startswith(prefix.lower()):
            return func(token)
    raise ValueError(f"Invalid token format: {token}. Must start with one of {list(literal_bit_funcs.keys())}")