"""Bounded, typed JSON wire values. No pickle, eval, or candidate code on host."""
import json
import math

MAX_ITEMS = 10000


def encode(value, depth=0):
    if depth > 32:
        raise ValueError("value nesting exceeds 32")
    t = type(value)
    if value is None:
        return ["none", None]
    if t is bool:
        return ["bool", value]
    if t is int:
        if value.bit_length() > 4096:
            raise ValueError("integer too large")
        return ["int", str(value)]
    if t is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return ["float", value]
    if t is str:
        if len(value) > 32768:
            raise ValueError("string too long")
        return ["str", value]
    if t in (list, tuple, set, frozenset):
        if len(value) > MAX_ITEMS:
            raise ValueError("too many items")
        items = [encode(x, depth+1) for x in value]
        if t in (set, frozenset):
            items.sort(key=lambda x: json.dumps(x, sort_keys=True))
        return [t.__name__, items]
    if t is dict:
        if len(value) > MAX_ITEMS:
            raise ValueError("too many items")
        items = [[encode(k, depth+1), encode(v, depth+1)] for k,v in value.items()]
        items.sort(key=lambda x: json.dumps(x[0], sort_keys=True))
        return ["dict", items]
    raise ValueError("unsupported return type: " + t.__name__)


def decode(wire, depth=0):
    if depth > 32 or type(wire) is not list or len(wire) != 2:
        raise ValueError("invalid wire value")
    tag, v = wire
    if tag == "none" and v is None:
        return None
    if tag == "bool" and type(v) is bool:
        return v
    if tag == "str" and type(v) is str and len(v) <= 32768:
        return v
    if tag == "int" and type(v) is str and len(v) <= 1235:
        return int(v)
    if tag == "float" and type(v) in (float, int) and math.isfinite(v):
        return float(v)
    if tag in ("list", "tuple", "set", "frozenset") and type(v) is list and len(v) <= MAX_ITEMS:
        values = [decode(x, depth+1) for x in v]
        return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[tag](values)
    if tag == "dict" and type(v) is list and len(v) <= MAX_ITEMS:
        return {decode(k, depth+1): decode(x, depth+1) for k,x in v}
    raise ValueError("unsupported wire value")


def equal(actual, expected):
    # Exact type-preserving comparison matches frozen integer/string/collection
    # cases; floating values use a declared tight tolerance recursively.
    a, b = decode(actual), decode(expected)
    def compare(x, y):
        if type(x) is not type(y):
            return False
        if type(x) is float:
            return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
        if type(x) in (list, tuple):
            return len(x) == len(y) and all(compare(i,j) for i,j in zip(x,y))
        if type(x) is dict:
            return x.keys() == y.keys() and all(compare(x[k], y[k]) for k in x)
        return x == y
    return compare(a,b)
