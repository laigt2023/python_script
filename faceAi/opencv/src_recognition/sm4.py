from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import binascii

def sm4_encrypt(key, plaintext):
    # 使用 PKCS7 填充
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext.encode('utf-8')) + padder.finalize()
    
    # 创建 SM4 加密器
    cipher = Cipher(algorithms.SM4(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # 加密数据
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
    return ciphertext.hex()

def sm4_decrypt(key, ciphertext):
    # 创建 SM4 解密器
    cipher = Cipher(algorithms.SM4(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    
    # 解密数据
    decrypted_data = decryptor.update(binascii.unhexlify(ciphertext)) + decryptor.finalize()
    
    # 去除填充
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(decrypted_data) + unpadder.finalize()
    
    return plaintext.decode('utf-8')

# 定义密钥和明文
key = bytes.fromhex("18fb264463a05a42" * 2)
plaintext = "44050619750510071X"

# 加密
encrypted_data = sm4_encrypt(key, plaintext)
print("Encrypted data:", encrypted_data)

# 解密
decrypted_data = sm4_decrypt(key, encrypted_data)
print("Decrypted data:", decrypted_data)