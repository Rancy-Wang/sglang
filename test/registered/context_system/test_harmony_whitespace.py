"""Native Harmony header discovery must not stall a shared HTTP event loop."""

import random
import re
import time

from test_ir import ROOT, load_file

harmony = load_file(
    "context_native_harmony",
    ROOT / "python/sglang/srt/parser/harmony_parser.py",
)


def test_header_discovery_preserves_native_strategy_selection():
    old = re.compile(
        r"(?:^|\s)(?:assistant)?\s*(analysis|commentary|assistantfinal)",
        re.IGNORECASE,
    )
    rng = random.Random(721)
    parts = [" ", "\n", "\t", "\u2003", "x", "Assistant", "analysis",
             "commentary", "assistantfinal", "assistantfin", "assistantanalysis"]
    for _ in range(1500):
        text = "".join(rng.choices(parts, k=rng.randrange(1, 10)))
        parser = harmony.HarmonyParser()
        parser.parse(text)
        assert isinstance(parser.strategy, harmony.TextStrategy) == bool(old.search(text))


def test_long_whitespace_stream_retains_partial_header_and_content():
    parser = harmony.HarmonyParser()
    whitespace = " \n\t" * 40_000
    start = time.monotonic()
    assert parser.parse(whitespace) == []
    # The old unanchored greedy prefix needs minutes on this input. This loose
    # bound detects that algorithmic regression, not normal parser speed noise.
    assert time.monotonic() - start < 2
    assert parser._buffer == whitespace
    assert parser.parse("Assistantanal") == []
    events = parser.parse("ysis hello")
    assert [(e.event_type, e.content) for e in events] == [("reasoning", " hello")]
    events = parser.parse(" assistantfinal done")
    assert [(e.event_type, e.content) for e in events] == [("normal", "done")]
