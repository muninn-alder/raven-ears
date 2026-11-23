"""Lead with the most informative details.

A voice card reaches a raven through a notification that gets clipped after a few
hundred characters, so the order of the lines decides what we hear in the moment.
How far she is from her usual voice, the peak, the moans and gasps, the laughs and
the stretches with no words come first; pitch numbers and loudness can be clipped.
Used by the sofa (every room, both ravens' cards) and the walkie.
"""
import re

ORDER = ["VS USUAL", "PEAK", "EVENTS", "😆 LAUGHS", "WORDLESS", "AFFECT", "SHAPE",
         "PITCH", "PACE", "BREATH", "SILENT", "SOUND", "DYN"]


def _rank(line):
    head = line.lstrip().upper()
    for i, label in enumerate(ORDER):
        if head.startswith(label):
            return i
    return len(ORDER)


def lead_lines(lines):
    """Same lines, juicy ones first. Unknown labels keep their order, after the known ones."""
    return sorted(lines, key=_rank)  # sorted() is stable


if __name__ == "__main__":
    import sys
    card = sys.stdin.read().strip()
    parts = re.split(r" · (?=[A-Za-z😆][A-Za-z ]*(?:\([^)]*\))?:)", card)
    print(" · ".join(lead_lines(parts)))
