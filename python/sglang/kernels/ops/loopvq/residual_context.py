# SPDX-License-Identifier: Apache-2.0
"""Increase only split scratch capacity; preserve the accepted partition policy."""


def context_source(source, max_context_len, *, grouped=False):
    if not 0 < max_context_len <= 40960:
        raise ValueError("VQ+ context must be within 40960 tokens")
    if max_context_len <= 32768:
        return source
    old_cap, new_cap = (512, 640) if grouped else (1024, 1280)
    anchor = f"CAP = {old_cap}, SPLITS ="
    if source.count(anchor) != 1:
        raise ValueError("Re-audit VQ+ context scratch declaration")
    source = source.replace(anchor, f"CAP = {new_cap}, SPLITS =")
    # Keep all <=32K partition choices unchanged, including the payload LUT
    # index. Above32K, the existing split clamp bounds per-split work to new_cap.
    variable = "longest" if grouped else "length"
    anchor = f"({variable} + CAP - 1) / CAP"
    if anchor not in source:
        raise ValueError("Re-audit VQ+ context partition formula")
    return source.replace(anchor, f"({variable} + {old_cap - 1}) / {old_cap}")


def context_suffix(max_context_len):
    return "_ctx40k" if max_context_len > 32768 else ""
