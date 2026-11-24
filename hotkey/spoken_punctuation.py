"""Turn spoken symbol names ("slash", "hyphen", "comma", "new line") into the symbols.

The ASR models write what they hear, so "src slash app" lands as words. This pass runs
on every transcript before it is pasted. Say "literal" before a word to keep it as a word
("literal slash" -> "slash"), or list phrases under spoken_punctuation.ignore in config.json.
"""

# How a symbol sits against its neighbours:
#   join  -> no space either side       (src/app, snake_case, example.com)
#   left  -> attaches to the word before (hello, world.)
#   right -> attaches to the word after  ((note, #tag, $5)
#   word  -> spaced like an ordinary word (a & b)
#   dash  -> joins both sides; a run of two or more keeps the space before (--help)
#   break -> line break, no space after
SYMBOLS = {
    'slash': ('/', 'join'),
    'forward slash': ('/', 'join'),
    'backslash': ('\\', 'join'),
    'back slash': ('\\', 'join'),
    'hyphen': ('-', 'dash'),
    'dash': ('-', 'dash'),
    'minus sign': ('-', 'dash'),
    'underscore': ('_', 'join'),
    'dot': ('.', 'join'),
    'at sign': ('@', 'join'),
    'tilde': ('~', 'right'),
    'comma': (',', 'left'),
    'period': ('.', 'left'),
    'full stop': ('.', 'left'),
    'question mark': ('?', 'left'),
    'exclamation mark': ('!', 'left'),
    'exclamation point': ('!', 'left'),
    'colon': (':', 'left'),
    'semicolon': (';', 'left'),
    'semi colon': (';', 'left'),
    'ellipsis': ('...', 'left'),
    'percent sign': ('%', 'left'),
    'open paren': ('(', 'right'),
    'open parenthesis': ('(', 'right'),
    'left paren': ('(', 'right'),
    'close paren': (')', 'left'),
    'close parenthesis': (')', 'left'),
    'right paren': (')', 'left'),
    'open bracket': ('[', 'right'),
    'close bracket': (']', 'left'),
    'open brace': ('{', 'right'),
    'open curly brace': ('{', 'right'),
    'close brace': ('}', 'left'),
    'close curly brace': ('}', 'left'),
    'hash sign': ('#', 'right'),
    'hashtag': ('#', 'right'),
    'dollar sign': ('$', 'right'),
    'ampersand': ('&', 'word'),
    'asterisk': ('*', 'word'),
    'plus sign': ('+', 'word'),
    'equals sign': ('=', 'word'),
    'new line': ('\n', 'break'),
    'newline': ('\n', 'break'),
    'new paragraph': ('\n\n', 'break'),
}
MAX_WORDS = max(len(phrase.split()) for phrase in SYMBOLS)
# Punctuation the ASR adds around a spoken symbol word ("src, slash, app.") is noise.
ASR_PUNCT = ',.;:!?'
ESCAPE = 'literal'
# "run the slash goal command" means "/goal", not "the/goal": after these words a
# path-like symbol (/ \\ .) starts a new word instead of joining the previous one.
STARTS_WORD_AFTER = {
    'a', 'an', 'the', 'my', 'our', 'your', 'this', 'that', 'its', 'their',
    'run', 'type', 'use', 'call', 'try', 'invoke', 'open', 'edit', 'see', 'check',
    'in', 'into', 'to', 'from', 'with', 'on', 'at', 'is', 'are', 'was', 'then', 'say',
}
STARTERS = {'/', '\\', '.'}


def _bare(token):
    return token.strip(ASR_PUNCT + '"\'').casefold()


def settings(config):
    """Return (enabled, ignored phrases) from the parsed config.json dict."""
    section = config.get('spoken_punctuation', {}) if isinstance(config, dict) else {}
    if not isinstance(section, dict):
        section = {}
    enabled = section.get('enabled', True) is not False
    ignore = section.get('ignore', [])
    ignored = {str(p).strip().casefold() for p in ignore} if isinstance(ignore, list) else set()
    return enabled, ignored


def apply(text, ignored=frozenset()):
    if not text:
        return text
    table = {p: v for p, v in SYMBOLS.items() if p not in ignored}
    tokens = text.split()
    bare = [_bare(t) for t in tokens]
    out = ''
    glue = False       # True: next item attaches with no space
    prev_dash = False  # previous item was a dash symbol
    prev_symbol = False  # previous item was a symbol we inserted (never strip it as ASR noise)

    def trim(out, chars=ASR_PUNCT):
        out = out.rstrip(' ')
        return out if prev_symbol else out.rstrip(chars)

    i = 0
    while i < len(tokens):
        if bare[i] == ESCAPE and i + 1 < len(tokens) and _match(bare, i + 1, table):
            # "literal slash" keeps the word.
            i += 1
            out, glue, prev_dash, prev_symbol = _word(out, glue, tokens[i]), False, False, False
            i += 1
            continue
        hit = _match(bare, i, table)
        if hit is None:
            out, glue, prev_dash, prev_symbol = _word(out, glue, tokens[i]), False, False, False
            i += 1
            continue
        length, (symbol, mode) = hit
        if mode == 'join' and symbol in STARTERS and not glue and _last_word(out) in STARTS_WORD_AFTER:
            out = out.rstrip(' ') + ' ' + symbol
            glue = True
        elif mode == 'join':
            out = trim(out) + symbol
            glue = True
        elif mode == 'dash':
            nxt = _match(bare, i + length, table)
            starts_run = not prev_dash and nxt is not None and nxt[1][1] == 'dash'
            if prev_dash or not out or glue:
                out += symbol
            elif starts_run:
                # "npm dash dash version" -> "npm --version"
                out = out.rstrip(' ') + ' ' + symbol
            else:
                out = trim(out) + symbol
            glue = True
        elif mode == 'left':
            out = trim(out) + symbol
            glue = False
        elif mode == 'right':
            out = (out + ' ' if out and not glue else out) + symbol
            glue = True
        elif mode == 'word':
            out = (out + ' ' if out and not glue else out) + symbol
            glue = False
        else:  # break
            out = trim(out, ',;:') + symbol
            glue = True
        prev_dash = mode == 'dash'
        prev_symbol = True
        i += length
    return out


def _last_word(out):
    words = out.split()
    return _bare(words[-1]) if words else ''


def _match(bare, i, table):
    for length in range(min(MAX_WORDS, len(bare) - i), 0, -1):
        phrase = ' '.join(bare[i:i + length])
        if phrase in table:
            return length, table[phrase]
    return None


def _word(out, glue, token):
    if out and not glue:
        out += ' '
    return out + token
