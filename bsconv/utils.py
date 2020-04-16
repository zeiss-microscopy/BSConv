def forceTwoTuple(x):
    """
    If `x` is a tuple or a list, return `x`, otherwise return `(x, x)`.
    """
    if isinstance(x, list):
        x = tuple(x)
    if not isinstance(x, tuple):
        x = (x, x)
    return x


def human_readable_int(x):
    """
    Transforms the integer `x` into a string containing thousands separators.
    """
    assert isinstance(x, int)
    in_str = str(x)
    digit_count = len(in_str)
    out_str = ""
    for (n_digit, digit) in enumerate(in_str):
        if (n_digit > 0) and (((digit_count - n_digit) % 3) == 0):
            out_str += ","
        out_str += digit
    return out_str
