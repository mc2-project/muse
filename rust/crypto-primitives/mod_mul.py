import numpy as np

def binary_repr(x):
    return list(reversed([int(e) for e in np.binary_repr(x)]))

class Reduction1:
    def __init__(self, modulus):
        self.p = modulus
        # Next power of 2
        self.k = p.bit_length()

        self.table = {}
        for l in range(self.k, self.k*2):
            self.table[l] = (2**l % self.p)

        """ TODO: Can further reduce table because can just shift by one
        l: 2k-1 2k-2 ... k
        r: (2^l mod p)
        """

    def reduce(self, x):
        if x < self.p:
            return x

        s = binary_repr(x)
        r = 0
        for i in reversed(range(self.k, x.bit_length())):
            if s[i] == 1:
                r += self.table[i]
        r += sum([s[j]*2**j for j in range(0, self.k)])

        while r >= self.p:
            r = r - self.p
        return r

class Reduction2:
    def __init__(self, modulus):
        self.p = modulus
        # Next power of 2
        self.k = self.p.bit_length()
        self.shift = 2
        self.table = {
            0 :  lambda x,y=0: (x*y + x) % 2,
            1 :  lambda x,y=0: (x*y + y) % 2,
            2 :  lambda x,y=0: (x*y) % 2,
            3 :  lambda x,y=0: (x*y + x) % 2,
            4 :  lambda x,y=0: (x*y + y) % 2,
        }
        #self.table = {
        #    0 :  lambda x,y=0: x, 
        #    1 :  lambda x,y=0: x, 
        #    2 :  lambda x,y=0: x,
        #    3 :  lambda x,y=0: x,
        #    4 :  lambda x,y=0: x,
        #    5 :  lambda x,y=0: x,
        #    6 :  lambda x,y=0: x,
        #    7 :  lambda x,y=0: x,
        #    8 :  lambda x,y=0: x,
        #    9 :  lambda x,y=0: x,
        #    10 : lambda x,y=0: x,
        #    11 : lambda x,y=0: x,
        #    12 : lambda x,y=0: x,
        #    13 : lambda x,y=0: x,
        #    14 : lambda x,y=0: x,
        #    15 : lambda x,y=0: x,
        #    16 : lambda x,y=0: x,
        #    17 : lambda x,y=0: x,
        #    18 : lambda x,y=0: x,
        #    19 : lambda x,y=0: x,
        #    20 : lambda x,y=0: x,
        #    21 : lambda x,y=0: x,
        #    22 : lambda x,y=0: x,
        #    23 : lambda x,y=0: x,
        #    24 : lambda x,y=0: x,
        #    25 : lambda x,y=0: x,
        #    26 : lambda x,y=0: x,
        #    27 : lambda x,y=0: x,
        #    28 : lambda x,y=0: x,
        #    29 : lambda x,y=0: x,
        #    30 : lambda x,y=0: x,
        #    31 : lambda x,y=0: x,
        #    32 : lambda x,y=0: x,
        #    33 : lambda x,y=0: x,
        #    34 : lambda x,y=0: x,
        #    35 : lambda x,y=0: x,
        #    36 : lambda x,y=0: x,
        #    37 : lambda x,y=0: 0,
        #    38 : lambda x,y=0: 0,
        #    39 : lambda x,y=0: 0,
        #    40 : lambda x,y=0: 0,
        #}
        #self.table = {
        #    0  :  lambda x,y=0: x,
        #    1  :  lambda x,y=0: x + y,
        #    2  :  lambda x,y=0: x*y + x + y,
        #    3  :  lambda x,y=0: x*y + x + y, 
        #    4  :  lambda x,y=0: x*y + x + y,
        #    5  :  lambda x,y=0: x*y + x + y,
        #    6  :  lambda x,y=0: x*y + x + y,
        #    7  :  lambda x,y=0: x*y + x + y,
        #    8  :  lambda x,y=0: x*y + x + y,
        #    9  :  lambda x,y=0: x*y + x + y,
        #    10 :  lambda x,y=0: x*y + x + y,
        #    11 :  lambda x,y=0: x*y + x + y,
        #    12 :  lambda x,y=0: x*y + x + y,
        #    13 :  lambda x,y=0: x*y + x + y,
        #    14 :  lambda x,y=0: x*y + x + y,
        #    15 :  lambda x,y=0: x*y + x + y,
        #    16 :  lambda x,y=0: x*y + x + y,
        #    17 :  lambda x,y=0: x*y + x + y,
        #    18 :  lambda x,y=0: x*y + x + y,
        #    19 :  lambda x,y=0: x*y + x + y,
        #    20 :  lambda x,y=0: x*y + x + y,
        #    21 :  lambda x,y=0: x*y + x + y,
        #    22 :  lambda x,y=0: x*y + x + y,
        #    23 :  lambda x,y=0: x*y + x + y,
        #    24 :  lambda x,y=0: x*y + x + y,
        #    25 :  lambda x,y=0: x*y + x + y,
        #    26 :  lambda x,y=0: x*y + x + y,
        #    27 :  lambda x,y=0: x*y + x + y,
        #    28 :  lambda x,y=0: x*y + x + y,
        #    29 :  lambda x,y=0: x*y + x + y,
        #    30 :  lambda x,y=0: x*y + x + y,
        #    31 :  lambda x,y=0: x*y + x + y,
        #    32 :  lambda x,y=0: x*y + x + y,
        #    33 :  lambda x,y=0: x*y + x + y,
        #    34 :  lambda x,y=0: x*y + x + y,
        #    35 :  lambda x,y=0: x*y + x + y,
        #    36 :  lambda x,y=0: x*y + x + y,
        #    37 :  lambda x,y=0: x*y + y,
        #    38 :  lambda x,y=0: x*y,
        #    39 :  lambda x,y=0: 0,
        #    40 :  lambda x,y=0: 0,
        #}

    def _split(self, x):
        bits = [int(b) for b in np.binary_repr(x, width=2*self.k)]
        split = [bits[i:(i+self.k)] for i in reversed(range(0, len(bits), self.k))]
        return split

    def _shift(self, x, shift):
        return x[:shift], x[shift:] + [0]*shift

    def _bits_to_num(self, x):
        return sum(x[-(i+1)]*2**i for i in range(0, len(x)))

    def _add_num(self, x, num):
        x_num = self._bits_to_num(x)
        bits = [int(b) for b in np.binary_repr(x_num+num, width=len(x)+1)]
        return bits[0], bits[1:]

    def _add_repr(self, x, y):
        x_num = self._bits_to_num(x)
        y_num = self._bits_to_num(y)
        bits = [int(b) for b in np.binary_repr(x_num+y_num, width=len(x)+1)]
        return bits[0], bits[1:]

    def _mod(self, x):
        neg_p = -self.p
        c, res = self._add_num(x, neg_p)
        if c == 1:
            return x[-self.k:]
        else:
            return res[-self.k:]

    def reduce(self, x):
        split = self._split(x)
        print("SPLIT: ", split, "\n")
        
        for i in range(len(split)-1, 0, -1):
            t = split[i]
            shifts = self.k
            while shifts > 0:
                if shifts < self.shift:
                    to_shift = shifts
                else:
                    to_shift = self.shift 

                c, t = self._shift(t, to_shift)
                to_add = [self.table[i](*reversed(c)) for i in reversed(range(self.k))]

                c, t = self._add_repr(t, to_add)
                t = self._mod([c] + t) 
                shifts -= to_shift
                print(f"Shift={shifts}, i={i}, carry={c}, added={to_add}: {t}")

            c, split[i-1] = self._add_repr(t, split[i-1])
            split[i-1] = self._mod([c] + split[i-1])
        # TODO: while
        out = self._mod(split[0])
        return self._bits_to_num(out)


def test_reduce(reducer):
    p = reducer.p
    possible_vals = range(64, 65)
    #possible_vals = range(0, (p-1)**2)
    for elem in possible_vals:
        print(f"For {elem} expected {elem%p} got {reducer.reduce(elem)}")
        assert(reducer.reduce(elem) == (elem % p))

if __name__ == '__main__':
    #t = Reduction2(2061584302081)
    t = Reduction2(23)
    test_reduce(t)
