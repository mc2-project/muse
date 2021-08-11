from sage.crypto.boolean_function import BooleanFunction
from numpy import binary_repr

prime = 2061584302081
#prime = 7340033
#prime = 31
bits = ceil(prime.log(2))
variables = 2

# Generate the table
table = list()
for i in range(2**variables):
    # Get bits for each key in truth table
    bit_repr = binary_repr(i, width=variables)
    # Compute corresponding value
    num = sum([int(b)*2**(bits+j) for j, b in enumerate(reversed(bit_repr))]) % prime
    table.append(num)
print(table)

# Split the table into `bits` table so that can turn each into ANF
tables = []
for _ in range(bits):
    tables.append([])
for num in table:
    bit_repr = reversed(binary_repr(num, width=bits))
    for i, bit in enumerate(bit_repr):
        tables[i].append(int(bit))
print(tables)

for i, table in enumerate(tables):
    B = BooleanFunction(table)
    print(i, ": ", B.algebraic_normal_form())
