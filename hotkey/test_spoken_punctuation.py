"""Spoken symbol words become symbols; ordinary words and opt-outs are left alone."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spoken_punctuation import apply, settings

CASES = [
    ('open src slash app slash main dot py', 'open src/app/main.py'),
    ('Run the slash goal command', 'Run the /goal command'),
    ('Type slash clear then slash goal', 'Type /clear then /goal'),
    ('use and slash or', 'use and/or'),
    ('Forward slash Users', '/Users'),
    ('C colon backslash temp', 'C:\\temp'),
    ('tilde slash code', '~/code'),
    ('edit the dot env file', 'edit the .env file'),
    ('state hyphen of hyphen the hyphen art', 'state-of-the-art'),
    ('Dash dash help', '--help'),
    ('run npm dash dash version', 'run npm --version'),
    ('Use snake underscore case', 'Use snake_case'),
    ('mail pranav at sign example dot com.', 'mail pranav@example.com.'),
    ('Hello comma how are you question mark', 'Hello, how are you?'),
    ('Hello, comma, world, period.', 'Hello, world.'),
    ('Src, slash, app.', 'Src/app.'),
    ('wait colon three things semicolon done exclamation mark', 'wait: three things; done!'),
    ('call foo open paren x close paren', 'call foo (x)'),
    ('first line new line second line', 'first line\nsecond line'),
    ('para one new paragraph para two', 'para one\n\npara two'),
    ('tag hashtag launch', 'tag #launch'),
    ('a ampersand b', 'a & b'),
    ('literal slash is a word', 'slash is a word'),
    ('nothing to change here', 'nothing to change here'),
    ('', ''),
]

failures = []
for spoken, expected in CASES:
    got = apply(spoken)
    if got != expected:
        failures.append(f'{spoken!r}: expected {expected!r}, got {got!r}')

ignored = settings({'spoken_punctuation': {'ignore': ['Period']}})[1]
if apply('a period of time', ignored) != 'a period of time':
    failures.append('ignore list did not keep "period"')
if settings({'spoken_punctuation': {'enabled': False}})[0]:
    failures.append('enabled: false was not honoured')
if not settings({})[0] or not settings({'spoken_punctuation': []})[0]:
    failures.append('missing or malformed section should default to enabled')

if failures:
    print('\n'.join(failures))
    sys.exit(1)
print(f'PASS: {len(CASES)} spoken-punctuation cases and config checks')
