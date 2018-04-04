'''
DES and 3DES implementation in numpy.

Examples:
python 3des.py --encrypt input.png cipher.3des 198241eb69d702280bc7d27868de15e27e167eb7741a2075
python 3des.py --decrypt cipher.3des output.png 198241eb69d702280bc7d27868de15e27e167eb7741a2075
'''

import sys
import string
import numpy as np
import random

# Building blocks
def xor(A,B):
    ''' input: bit array A and B, output: A xor B  '''
    return 1*np.logical_xor(A,B)

def left_shift(arr,n):
    ''' performs circular left shift to bit array '''
    return np.roll(arr,-n)

def right_shift(arr,n):
    ''' performs circular right shift to bit array '''
    return np.roll(arr,n)

def permutation(x,table):
    ''' permute x -> y, where len(y)=len(table) '''
    y = np.zeros(len(table),dtype=int)
    for output_index, input_index in enumerate(table):
        y[output_index] = x[input_index]
    return y

def bit_array2decimal(bit_arr):
    ''' converts bit array to decimal number '''
    bit_arr = np.flip(bit_arr,axis=0)
    power_arr = 2**(np.arange( len(bit_arr) ))
    return (power_arr * bit_arr).sum()

def decimal2bit_array(decimal,width=4):
    ''' converts decimal number to 4 (or width) bits '''
    bin_str = np.binary_repr(decimal, width=width)
    return np.array([ int(bit) for bit in bin_str ],dtype=int)

# Tables
PC1_table = np.array([
    57, 49, 41, 33, 25, 17,  9,
     1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27,
    19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
     7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29,
    21, 13,  5, 28, 20, 12,  4
])
PC1_table -= 1
PC2_table = np.array([
    14, 17, 11, 24,  1,  5,  3, 28,
    15,  6, 21, 10, 23, 19, 12,  4,
    26,  8, 16,  7, 27, 20, 13,  2,
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32
])
PC2_table -= 1
S_table = np.zeros((8,4,16),dtype=int)
S_table[0] = np.array([
           [14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7],
           [ 0,  15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3, 8],
           [ 4,   1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5, 0],
           [15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13]
])
S_table[1] = np.array([
           [15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10],
           [ 3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5],
           [ 0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15],
           [13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9]
])
S_table[2] = np.array([
           [10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8],
           [13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1],
           [13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7],
           [ 1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12]
])
S_table[3] = np.array([
           [ 7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15],
           [13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9],
           [10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4],
           [ 3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14]
])
S_table[4] = np.array([
           [ 2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9],
           [14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6],
           [ 4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14],
           [11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3]
])
S_table[5] = np.array([
           [12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11],
           [10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8],
           [ 9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6],
           [ 4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13]
])
S_table[6] = np.array([
           [ 4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1],
           [13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6],
           [ 1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2],
           [ 6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12]
])
S_table[7] = np.array([
           [13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7],
           [ 1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2],
           [ 7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8],
           [ 2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11]
])
P_table = np.array([
    16,  7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26,  5, 18, 31, 10,
     2,  8, 24, 14, 32, 27,  3,  9,
    19, 13, 30,  6, 22, 11,  4, 25
])
P_table -= 1
E_table = np.array([
    32,  1,  2,  3,  4,  5,
     4,  5,  6,  7,  8,  9,
     8,  9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32,  1
])
E_table -= 1
IP_table = np.array([
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
])
IP_table -= 1
FP_table = np.array([
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41,  9, 49, 17, 57, 25
])
FP_table -= 1

# Key schedule
def init_transform(k):
    ''' Applies PC1 permutation. output: 2*(28,) '''
    k = permutation(k,PC1_table)
    return k[:28],k[28:]

def transform(C,D,i,shift=left_shift):
    ''' C,D=(28,), i=round number, n=how many bits to rotate '''
    n = 1 if i in [0,1,8,15] else 2
    C, D = shift(C,n), shift(D,n)
    C_and_D = np.concatenate((C,D))
    k = permutation(C_and_D,PC2_table)
    return C,D,k

def derive_keys(Key):
    ''' Derives 16 keys of 48 bit length out of Key(64 len) '''
    k = np.zeros((16,48),dtype=int)
    C,D = init_transform(Key)
    for i in range(16):
        C,D,k[i] = transform(C,D,i)
    return k

# DES S-Boxes
def S(x,i):
    ''' Applies S function. input: 6 bits output: 4 bits '''
    row_index_bin = [ bit for i,bit in enumerate(x) if i in [0,5] ]
    column_index_bin = [ bit for i,bit in enumerate(x) if i in [1,2,3,4] ]
    dec_val = S_table[ i, bit_array2decimal(row_index_bin), bit_array2decimal(column_index_bin) ]
    return decimal2bit_array(dec_val)

# f-Function
def P(x):
    return permutation(x,P_table)

def E(x):
    return permutation(x,E_table)

def f(R,k):
    ''' Applies f function to 32-bit input R with k[i] from key schedule '''
    y = E(R)
    y = xor(y,k)
    y = np.split(y,8)
    for i in range(8):
        y[i] = S(y[i],i)
    y = np.concatenate(y)
    return P(y)

# DES Round
def Round(L,R,k,f=f):
    ''' Applies DES Round to 32-bit halves L,R '''
    return R, xor(L,f(R,k))

# Initial and Final permutations
def IP(x):
    return permutation(x,IP_table)

def FP(x):
    return permutation(x,FP_table)

# Block Encryption and Decryption
def encrypt(x,key):
    ''' Encrypts x 64-bit block '''
    x = IP(x)
    k = derive_keys(key)
    L,R = np.split(x,2)
    for i in range(16):
        L,R = Round(L,R,k[i])
    x = np.concatenate((R,L))
    y = FP(x)
    return y

def decrypt(y,key):
    ''' Decrypts y 64-bit block '''
    y = IP(y)
    k = derive_keys(key)
    L,R = np.split(y,2)
    for i in range(16)[::-1]:
         L,R = Round(L,R,k[i])
    y = np.concatenate((R,L))
    x = FP(y)
    return x

# File reading
def read_file(file_name):
    ''' Read file, return array of bits '''
    file_in_bytes = np.fromfile(file_name, dtype = "uint8")
    file_in_bits = np.unpackbits(file_in_bytes)
    return file_in_bits

def write_file(file_name,file_in_bits):
    ''' Take bit array and write it to file in byte form '''
    file_in_bytes = np.packbits(file_in_bits)
    file_in_bytes.tofile(file_name)

# Padding
def pad_file(file_in_bits):
    ''' Adds PKCS5 padding '''
    n = file_in_bits.size%64
    N = int( 8 - np.ceil(n/8) )
    for i in range(N):
        byte_in_bits = decimal2bit_array(N,width=8)
        file_in_bits = np.append( file_in_bits, byte_in_bits, axis=0 )
    return file_in_bits

def unpad_file(file_in_bits):
    ''' Removes PKCS5 padding '''
    last_byte_in_bits = file_in_bits[-8:]
    l = bit_array2decimal(last_byte_in_bits)
    n = file_in_bits.size
    return np.resize(file_in_bits,n-(l*8))

# File Encryption and Decryption
def encrypt_file(file_in_bits,key):
    ''' Returns full encrypted file '''
    no_blocks = int(file_in_bits.size/64)
    blocks = file_in_bits.reshape((no_blocks,64))
    encrypted_file = []
    for i in range(no_blocks):
        x = blocks[i]
        y = encrypt(x,key)
        encrypted_file.extend(y)
        print('Encrypting blocks: {}/{} '.format(i, no_blocks), end='\r')
    print('')
    return np.array(encrypted_file)

def decrypt_file(file_in_bits,key):
    ''' Returns full decrypted file '''
    no_blocks = int(file_in_bits.size/64)
    blocks = file_in_bits.reshape((no_blocks,64))
    decrypted_file = []
    for i in range(no_blocks):
        y = blocks[i]
        x = decrypt(y,key)
        decrypted_file.extend(x)
        print('Decrypting blocks: {}/{} '.format(i, no_blocks), end='\r')
    print('')
    return np.array(decrypted_file)

# DES and 3DES
def des_enc(input_name,output_name,key):
    ''' Saves encrypted file to "output_name" file '''
    file_in_bits = read_file(input_name)
    file_in_bits = pad_file(file_in_bits)
    encrypted_file = encrypt_file(file_in_bits,key)
    write_file(output_name,encrypted_file)
    
def des_dec(input_name,output_name,key):
    ''' Saves decrypted file to "output_name" file '''
    file_in_bits = read_file(input_name)
    decrypted_file = decrypt_file(file_in_bits,key)
    decrypted_file = unpad_file(decrypted_file)
    write_file(output_name,decrypted_file)

def tri_des_enc(input_name,output_name,key):
    ''' Saves encrypted file to "output_name" file '''
    key = key.reshape((3,64))
    file_in_bits = read_file(input_name)
    file_in_bits = pad_file(file_in_bits)
    encrypted_file = file_in_bits
    for i in range(3):
        encrypted_file = encrypt_file(encrypted_file,key[i])
    write_file(output_name,encrypted_file)

def tri_des_dec(input_name,output_name,key):
    ''' Saves decrypted file to "output_name" file '''
    key = key.reshape((3,64))
    file_in_bits = read_file(input_name)
    decrypted_file = file_in_bits
    for i in range(3)[::-1]:
        decrypted_file = decrypt_file(decrypted_file,key[i])
    decrypted_file = unpad_file(decrypted_file)
    write_file(output_name,decrypted_file)

if __name__ == '__main__':
    if sys.argv[1] == '--encrypt' and len(sys.argv) == 5:
        input_filename = sys.argv[2]
        output_filename = sys.argv[3]
        key = sys.argv[4]
        # if len(key) != 192 -> generate random one
        if len(key.encode('utf8'))*4 != 192:
            key = ''.join([random.SystemRandom().choice('0123456789abcdef') for _ in range(48)])
            print('Key must be 48 hex digit string. Generated new key:', key)
        # convert key to bit array
        key_in_bits = []
        for dec_val in [ int(hex_char,16) for hex_char in key ]:
            key_in_bits.append( decimal2bit_array(dec_val) )
        key_in_bits = np.concatenate(key_in_bits)
        # apply encryption
        tri_des_enc(input_filename,output_filename,key_in_bits)
        print('Encryption is done!')

    elif sys.argv[1] == '--decrypt' and len(sys.argv) == 5:
        input_filename = sys.argv[2]
        output_filename = sys.argv[3]
        key = sys.argv[4]
        # if len(key) != 192 -> raise error
        if len(key.encode('utf8'))*4 != 192:
            raise SyntaxError('Wrong key. Key must be 48 hex digit string.')
        # convert key to bit array
        key_in_bits = []
        for dec_val in [ int(hex_char,16) for hex_char in key ]:
            key_in_bits.append( decimal2bit_array(dec_val) )
        key_in_bits = np.concatenate(key_in_bits)
        # apply decryption
        tri_des_dec(input_filename,output_filename,key_in_bits)
        print('Decryption is done!')
    else:
        print('Invalid arguments. Try:')
        print('--encrypt [input_filename] [output_filename] [48-hex digit key]')
        print('--decrypt [input_filename] [output_filename] [48-hex digit key]')