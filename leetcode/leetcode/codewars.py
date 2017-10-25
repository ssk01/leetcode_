def namelist(names):
    #your code here
    l = len(names)
    if l == 0: return ''
    if l == 1:
        return names[0]['name']
    if l == 2:
        res = ''
        res += names[0]['name']
        res +=' & '
        res += names[1]['name']
        return res
    res = ''
    for i in range(l-2):
        res += names[i]['name']
        res += ', '
    res += names[l-2]['name']
    res +=' & '
    res += names[1-1]['name']
    return res

class HighScoreTable:
    def __init__(self,n):
        # YOUR CODE HERE
        self.i = 0
        self.n = n
        self.scores = []
    def update(self, val):
        # YOUR CODE HERE
        i += 1
        self.scores.append(val)
        self.scores.sort()
        if i > n:
            self.scores = self.scores[:n]

    def reset(self):
        # YOUR CODE HERE
        self.scores = []
import re

def tokenize(expression):
    if expression == "":
        return []

    regex = re.compile("\s*(=>|[-+*\/\%=\(\)]|[A-Za-z_][A-Za-z0-9_]*|[0-9]*\.?[0-9]+)\s*")
    tokens = regex.findall(expression)
    return [s for s in tokens if not s.isspace()]

class Interpreter:
    def __init__(self):
        self.vars = {}
        self.functions = {}

    def input(self, expression):
        tokens = tokenize(expression)
        print(tokens)

Interpreter = Interpreter()
Interpreter.input("21231 - 1")
Interpreter.input("2.342 * 3")
Interpreter.input("8 / 4")
Interpreter.input("x1 = 1")
Interpreter.input("x_3")