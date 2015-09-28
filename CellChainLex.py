from ply import lex

__author__ = 'mfansler'

# Tokens
tokens = (
    'PARTIAL', 'DELTA',
    'GROUP', 'IDENTIFIER',
    'PLUS', 'EQUALS', 'OTIMES',
    'LBRACE', 'RBRACE',
    'LPAREN', 'RPAREN',
    'COMMA'
)

# Token definitions
t_PARTIAL    = r'\\partial'
t_DELTA      = r'\\Delta'
t_OTIMES      = r'\\otimes'
t_IDENTIFIER = r'[a-zA-Z0-9]+(_{[a-zA-Z0-9]+})?'
t_PLUS       = r'\+'
t_EQUALS     = r'='
t_LBRACE     = r'\\{'
t_RBRACE     = r'\\}'
t_LPAREN     = r'\('
t_RPAREN     = r'\)'
t_COMMA      = r','

# Ignored characters
t_ignore = " \t"

def t_GROUP(t):
    r'C_{([0-9]+)}\([a-zA-Z0-9]*\)'
    t.value = int(t.lexer.lexmatch.group(2))
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

# Ignore comment lines
def t_comment(t):
    r'%.*'

def t_error(t):
    print "Illegal character '%s'" % t.value[0]
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

if __name__ == "__main__":
    lex.runmain(lexer)