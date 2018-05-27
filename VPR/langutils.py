import re, os
from pymystem3 import Mystem
import re

bad_punct = {' + ', ' - ', ' & ', ': ', ' ** ', ' * ', ' / ', ' - ', ',', ' - - ',
             ',/ ', ' :', ' / / ', ' – ', ')', '(', '[', ']', '"', "'", '\t', '  ', '\u00AB', '\u00BB'}

bp_single_ch = {ch for ch in bad_punct if len(ch) == 1}
bp_table = str.maketrans(''.join(bp_single_ch), ' ' * len(bp_single_ch))
bp_many_ch = bad_punct - bp_single_ch


def replace_punct(s, more='?|\n'):
    s = str(s).lower().translate(bp_table)
    for punct in bp_many_ch:
        s = s.replace(punct, ' ')
    for p in more:
        s = s.replace(p, ' ')
    return s


eng_layout = '`qwertyuiop[]asdfghjkl;\'zxcvbnm,./'
rus_layout = 'ёйцукенгшщзхъфывапролджэячсмитьбю.'
layout_table = str.maketrans(eng_layout + rus_layout,
                             rus_layout + eng_layout)


def switch_layout(word):
    return word.translate(layout_table)


symb_eng_str = 'qwertyuiopasdfghjklzxcvbnm'
symb_rus_str = 'ёйцукенгшщзхъфывапролджэячсмитьбю'
symb_eng_set = set(symb_eng_str)
symb_rus_set = set(symb_rus_str)
sim_eng_str = 'xcaopeykm'
sim_rus_str = 'хсаореукм'
diff_rus_set = symb_rus_set - set(sim_rus_str)
diff_eng_set = (symb_eng_set - set(sim_eng_str)) | set(list('1234567890'))
sim_symb_enru = str.maketrans(sim_eng_str, sim_rus_str)
sim_symb_ruen = str.maketrans(sim_rus_str, sim_eng_str)

query_bad_chars = '$,\'";:|\t&![]()/§'
query_table = str.maketrans(query_bad_chars, ' ' * len(query_bad_chars))


def split_clear_query(s):
    s = str(s).lower().translate(query_table)
    return [word for word in s.split(' ') if len(word)]


def check_and_touch_file(path):
    if not os.path.exists(path):
        open(path, 'w').close()
        os.chmod(path, 0o666)


def normal_form(word, m):
    analyze = m.analyze(word)[0]["analysis"]
    if len(analyze) == 0:
        return ''
    if analyze[0]['gr'][0] == 'S':
        return analyze[0]['lex']
    else:
        return ''


def sentence_to_normal(words, m):
    lem = m.lemmatize(words)
    if len(lem) == 0:
        return words
    return ''.join(lem[:-1])


def all_eng(st: str):
    return len(set(st) - symb_eng_set) == 0


def is_word(st: str):
    ''' Returns True if `^[а-яА-ЯёЁa-zA-Z]+[-]*[а-яА-ЯёЁa-zA-Z]+$` is matched.'''
    return len(st) >= 2 and st[0] != '-' and st[-1] != '-' and \
           len(set(st.lower()) - set('-')  - set("'")  - symb_rus_set - symb_eng_set) == 0


def can_be_switched(st: str):
    '''Retuns True if st or st with switches layout are alphabetic (,./[])'''
    return len(st) >= 2 and st[0] != '-' and st[-1] != '-' and \
           len(set(st.lower()) - set(list("'-[]./,")) - symb_rus_set - symb_eng_set) == 0


def change_sim_symb(st: str):
    st_diff_charset = set(st) & (diff_rus_set | diff_eng_set)
    if len(st_diff_charset):  # if there some unique chars
        # and if there are no unique english chars
        if not len(st_diff_charset & diff_eng_set):
            st = st.translate(sim_symb_enru)
        # or if there are no unique russian
        elif not len(st_diff_charset & diff_rus_set):
            st = st.translate(sim_symb_ruen)
    return st


def samelang(w1: str, w2: str):
    w1_rus = len(set(w1.lower()) - symb_rus_set) > 0
    w2_rus = len(set(w2.lower()) - symb_rus_set) > 0
    return w1_rus == w2_rus


def dividebylang(query: str):
    rus = symb_rus_set
    # ToDo: that's what we need?
    any_lang = '1234567890-\/,'
    stop = ' '
    res, pos = '', 0
    skip = True
    for i, c in enumerate(query):
        if skip:
            skip = False
            # if it's still a stop-char or maybe
            if c in stop:  # there were no sign of specific language
                skip = True
                res += query[pos:i]
                pos = i
            # if it's the beginning of the word, we want to know it's lang
            else:
                state = c in rus
            skip |= c in any_lang  # yet there is a chance we can't
            continue
        if c in stop:  # if we encounter stop-char after some specific language
            # we replace it with space
            res += query[pos:i] + ' '
            # we'll skip next condition for the first char of the next word
            #                        or for another stop-char
            pos, skip = i + 1, True
            continue
        elif state ^ (c in rus) and not c in any_lang:
            # if lang changed in the middle of the word
            # we'll divide two parts by space
            res += query[pos:i] + ' '
            pos, state = i, c in rus

    res += query[pos:]
    return res


def strip_left(word, validator):
    '''
        Recursive function that tries to cut provided string in valid words from left to the right.
        Only if all resulting words are valid, list of the words will be returned.
        Second returned value is confidence in returned words.
        (Should be all 1 when used with basic validator)
    '''
    w, c = '', []
    word, w_c = validator(word)
    if w_c:
        return [word], [w_c]
    elif len(word) >= 6:
        for pivot in range(4, len(word) - 3):
            # ^ we decided that ~valid~ word is longer than 3 chars
            left_w, left_c = validator(word[:pivot])
            if left_c:
                right_w, right_c = strip_left(word[pivot:], validator)
                if all(right_c) and len(left_w):
                    return [left_w] + right_w, [left_c] + right_c
                    #    if (pivot <= 7 and word[pivot-1] == 'о')\
                    #            or (pivot <= 4 and not word[pivot-1] == 'а') or right_c:
                    #            return [left_w] + [right_w[0]] + right_w[1:],\
                    #               [left_c+ right_c[0]] + right_c[1:]
                    #    else: # not `электро`, `радио` and not `авиа`, `аква`, etc.
                    #       return [left_w] + right_w[1:],\
                    #               [left_c] + right_c[1:]
    # in case we didn't find anything
    return '', [False]


def split_set_of_words(words):
    ans = set()
    for word in words:
        ans.update(word.split('-'))
    return ans


def del_comma(word: str):
    word = word.strip(',')
    word_clear = ''
    for i in range(len(word)):
        if word[i] != ',' or (word[i] == ',' and word[i + 1].isdigit() and word[i - 1].isdigit()):
            word_clear += word[i]
        else:
            word_clear += ' '
    word = word_clear
    return word


def replace_punctuation_in_searchstring(searchstring: str):
    new_ss = searchstring.lower()
    stoptrans = str.maketrans('', '', '?!#$%^*|=+~')
    new_ss = new_ss.translate(stoptrans)  # built-in for the win
    if new_ss.count("'") % 2 == 0:
        new_ss = new_ss.replace("'", ' ')
    if new_ss.count('"') % 2 == 0:
        new_ss = new_ss.replace('"', ' ')
    return new_ss
