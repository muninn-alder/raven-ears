"""The card leaves the feeling to its listener. Shared, formatting-only hooks."""

HANDOFF = 'and now, you'


def finish_lines(lines):
    """Return a copy with exactly one terminal handoff; preserve the card's contents."""
    out = list(lines)
    while out and (not out[-1].strip() or out[-1] == HANDOFF):
        out.pop()
    return out + [HANDOFF]


def finish_json(data):
    """Keep structured modes valid JSON; the handoff is their final field."""
    return {key: value for key, value in data.items() if key != 'listener'} | {'listener': HANDOFF}
