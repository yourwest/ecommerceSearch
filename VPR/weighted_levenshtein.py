import numpy as np
from weighted_levenshtein import lev, osa, dam_lev
import itertools
from vypryamitel.langutils import switch_layout, symb_eng_set


def validstr(w):
    return all([ord(c) < 127 for c in w])


trans = dict(zip(list('йцукенгшщзхъфывапролджэячсмитьбюё'), [chr(i) for i in range(32, 65)]))
groups = [list('аоуыя'), list('иеюяэ'), list('бп'), list('вф'), list('дт'),
          list('гкх'), list('зс'), list('чщ'), list('жш'), list('ёо'),
          list('vw'), list('sc'), list('o0'), list('nm'), list('- '),
          list('l1'), list('aeo'), list('jg'), list('wv'), list('sz')]

for i in range(len(groups)):
    for j in range(len(groups[i])):
        groups[i][j] = trans.get(groups[i][j], groups[i][j])

keyboard = [('й', list('цыф')),
            ('ц', list('уыфй')),
            ('у', list('квыц')),
            ('к', list('еаву')),
            ('е', list('нпак')),
            ('н', list('грпе')),
            ('г', list('шорн')),
            ('ш', list('щлог')),
            ('щ', list('здлш')),
            ('з', list('хждщ')),
            ('ъ', list('эх')),
            ('ф', list('йцыя')),
            ('ы', list('йцувчяф')),
            ('в', list('цукасчы')),
            ('а', list('укепмсв')),
            ('п', list('кенрима')),
            ('р', list('енготип')),
            ('о', list('нгшльтр')),
            ('л', list('гшщдбьо')),
            ('д', list('шщзжюбл')),
            ('ж', list('щзхэюд')),
            ('э', list('жзхъ')),
            ('я', list('фыч')),
            ('ч', list('яывс')),
            ('с', list('чвам')),
            ('м', list('сапи')),
            ('и', list('мпрт')),
            ('т', list('ироь')),
            ('ь', list('толб')),
            ('б', list('ьлдю')),
            ('ю', list('бдж'))
            ]

keyboard_eng = [(switch_layout(a), list(switch_layout(''.join(b)))) for a, b in keyboard]
keyboard += keyboard_eng

keyboard_l = [[a, n] for a, n in keyboard]
for i in range(len(keyboard_l)):

    keyboard_l[i][0] = trans.get(keyboard_l[i][0], keyboard_l[i][0])
    for j in range(len(keyboard_l[i][1])):
        keyboard_l[i][1][j] = trans.get(keyboard_l[i][1][j], keyboard_l[i][1][j])

keyboard = [(el[0], el[1]) for el in keyboard_l]

insert_costs = np.ones(128, dtype=np.float64)
delete_costs = np.ones(128, dtype=np.float64)
substitute_costs = np.ones((128, 128), dtype=np.float64)
transpose_costs = np.ones((128, 128), dtype=np.float64) / 4

for e in list('ъэ'):
    delete_costs[ord(trans.get(e, e))] = 0.5

for gr in groups:
    for a, b in itertools.combinations(gr, 2):
        substitute_costs[ord(a), ord(b)] = 0.5
        substitute_costs[ord(b), ord(a)] = 0.5

for a, neibs in keyboard:
    for n in neibs:
        substitute_costs[ord(a), ord(n)] = 0.5
        substitute_costs[ord(n), ord(a)] = 0.5


def my_weighted_levenshtein(s1, s2):
    st1 = ''.join([trans.get(s, s) for s in s1])
    st2 = ''.join([trans.get(s, s) for s in s2])
    if validstr(st2):
        return dam_lev(st1.encode('utf-8'), st2.encode('utf-8'), insert_costs=insert_costs, delete_costs=delete_costs,
                       substitute_costs=substitute_costs, transpose_costs=transpose_costs)
    else:
        return len(st1) + 1
