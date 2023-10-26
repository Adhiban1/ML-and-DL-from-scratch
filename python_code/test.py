from time import time
import os

print('Python...')
start = time()
os.system('python3 main.py')
python = time() - start
print()

print('Rust...')
start = time()
os.system('./target/release/rust')
rust = time() - start
print()

if rust < python:
    print('Rust wins')
    print(f'Rust is {python/rust:.3} times faster than Python')
else:
    print('Python wins')
    print(f'Python is {rust/python:.3} times faster than Rust')

'''
$ python3 test.py 
Python...
Inital loss: 12.91907527386166
Lowest loss: 0.00025037345234715043
Final loss: 0.00025037345234715043

Rust...
Inital loss: 14.449090232922213
Lowest loss: 0.000005498838681749966
Final loss: 0.000005498838681749966

Rust wins
Rust is 19.8 times faster than Python
'''