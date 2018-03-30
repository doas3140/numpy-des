# Basic implementation of DES and 3DES with numpy

Created for educational purposes. Performance is slow.

## Commands

Commands:

Encrypt: `python 3des.py --encrypt [input_filename] [output_filename] [48-hex digit key]`
Decrypt: `python 3des.py --decrypt [input_filename] [output_filename] [48-hex digit key]`

Examples:

Encrypt: `python 3des.py --encrypt input.png cipher.3des 198241eb69d702280bc7d27868de15e27e167eb7741a2075`
Decrypt: `python 3des.py --decrypt cipher.3des output.png 198241eb69d702280bc7d27868de15e27e167eb7741a2075`