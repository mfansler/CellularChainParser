from ply import yacc

import CellChainLex

__author__ = 'mfansler'

# Get the token map
tokens = CellChainLex.tokens

def p_program(p):
    '''program : group_list differential_list coproduct_list
               | group_list differential_list'''
    p[0] = {"groups": p[1], "differentials": p[2], "coproducts": p[3] if len(p) > 3 else list()}

def p_group_list_1(p):
    "group_list : group_list statement_group"
    p[0] = p[1]
    p[0].update(p[2])

def p_group_list_2(p):
    "group_list : statement_group"
    p[0] = p[1]

def p_differential_list_1(p):
    "differential_list : differential_list statement_differential"
    p[0] = p[1]
    p[0].update(p[2])

def p_differential_list_2(p):
    "differential_list : statement_differential"
    p[0] = p[1]

def p_coproduct_list_1(p):
    "coproduct_list : coproduct_list statement_coproduct"
    p[0] = p[1]
    p[0].update(p[2])

def p_coproduct_list_2(p):
    "coproduct_list : statement_coproduct"
    p[0] = p[1]

def p_statement_group(p):
    "statement_group : GROUP EQUALS LBRACE identifier_list RBRACE"
    p[0] = {p[1]: p[4]}

def p_statement_differential(p):
    "statement_differential : PARTIAL IDENTIFIER EQUALS expression"
    p[0] = {p[2]: p[4]}

def p_statement_coproduct(p):
    "statement_coproduct : DELTA IDENTIFIER EQUALS expression"
    p[0] = {p[2]: p[4]}

def p_identifier_list_1(p):
    "identifier_list : IDENTIFIER"
    p[0] = list()
    p[0].append(p[1])

def p_identifier_list_2(p):
    "identifier_list : identifier_list COMMA IDENTIFIER"
    p[0] = p[1]
    p[0].append(p[3])

def p_expression_sum_1(p):
    "expression : expression PLUS IDENTIFIER"
    p[0] = p[1]
    if p[3] in p[0]:
        p[0][p[3]] += 1
    else:
        p[0][p[3]] = 1

def p_expression_sum_2(p):
    "expression : expression PLUS IDENTIFIER OTIMES IDENTIFIER"
    p[0] = p[1]
    if (p[3], p[5]) in p[0]:
        p[0][(p[3], p[5])] += 1
    else:
        p[0][(p[3], p[5])] = 1

def p_expression_sum_distribute_right(p):
    "expression : expression PLUS IDENTIFIER OTIMES LPAREN expression RPAREN"
    p[0] = p[1]
    for key, value in p[6].iteritems():
        if (p[3], key) in p[0]:
            p[0][(p[3], key)] += value
        else:
            p[0][(p[3], key)] = value

def p_expression_sum_distribute_left(p):
    "expression : expression PLUS LPAREN expression RPAREN OTIMES IDENTIFIER"
    p[0] = p[1]
    for key, value in p[4].iteritems():
        if (key, p[7]) in p[0]:
            p[0][(key, p[7])] += value
        else:
            p[0][(key, p[7])] = value

def p_expression_product(p):
    "expression : IDENTIFIER OTIMES IDENTIFIER"
    p[0] = {(p[1], p[3]): 1}

def p_expression_product_distribute_right(p):
    "expression : IDENTIFIER OTIMES LPAREN expression RPAREN"
    p[0] = {}
    for key, value in p[4].iteritems():
        p[0][(p[1], key)] = value

def p_expression_product_distribute_left(p):
    "expression : LPAREN expression RPAREN OTIMES IDENTIFIER"
    p[0] = {}
    for key, value in p[2].iteritems():
        p[0][(key, p[5])] = value

def p_expression_product_distribute_both(p):
    "expression : LPAREN expression RPAREN OTIMES LPAREN expression RPAREN "
    p[0] = {}
    for l_key, l_value in p[2].iteritems():
        for r_key, r_value in p[6]:
            p[0][(l_key, r_key)] = l_value*r_value

def p_expression_identifier(p):
    "expression : IDENTIFIER"
    p[0] = {p[1]: 1}

def p_error(p):
    if p:
        print "Syntax error at '%s' on line %d" % (p.value, p.lineno)
    else:
        print "Syntax error at EOF"

# Build the parser
cell_chain_parser = yacc.yacc()

def parse(data, debug=0):
    cell_chain_parser.error = 0
    p = cell_chain_parser.parse(data, debug=debug)
    if cell_chain_parser.error:
        return None
    return p

if __name__ == "__main__":
    while 1:
        try:
            s = raw_input()
        except EOFError:
            break
        if not s:
            continue
        parse(s)
