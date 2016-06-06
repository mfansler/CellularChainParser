from re import sub

# chars
DELTA   = u"\u0394"
PARTIAL = u"\u2202"
NABLA   = u"\u2207"
OTIMES  = u"\u2297"
THETA   = u"\u03b8"
PHI     = u"\u03c6"
CHAINPARTIAL = "(1" + OTIMES + PARTIAL + " + " + PARTIAL + OTIMES + "1)"


def format_cells(cells):
    return sub(',', '_', sub(r'[{}]', '', str(cells)))


def format_tuple(t):
    if type(t) is tuple:
        return u" \u2297 ".join([format_sum(x) if type(x) is list else x for x in t])
    else:
        return unicode(t)


def format_sum(obj):
    if obj is None:
        return "0"
    elif type(obj) is dict:
        single = [format_tuple(k) for k, v in obj.items() if v == 1]
        multiple = [u"{}*({})".format(v, format_tuple(k)) for k, v in obj.items() if v > 1]
        return u" + ".join(single + multiple)
    elif type(obj) is list:
        return u"(" + u" + ".join([format_tuple(o) for o in obj]) + ")"
    else:
        return obj


def format_morphism(m):
    formatted_maps = []
    for k, v in m.iteritems():
        if v:
            if type(v) is dict:
                formatted_v = '(' + format_morphism(v) + ')'
            else:
                formatted_v = format_sum(v)
            formatted_maps.append(u"{}{}_{{{}}}".format(formatted_v, PARTIAL, k))

    return u"\n\t+ ".join(formatted_maps)
