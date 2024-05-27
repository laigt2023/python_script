from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

def sm3_encrypt(message):
    # Create a hash context using SM3 algorithm
    digest = hashes.Hash(hashes.SM3(), backend=default_backend())
    # Update the context with the message
    digest.update(message.encode('utf-8'))
    # Finalize the hash and get the digest
    hashed_message = digest.finalize()
    # Convert the digest to hexadecimal representation
    return hashed_message.hex()


# 待加密的数据
data = "accessNo=e85c946c15f0338a473a9e75dfc6e17e&factoryNum=4e9d4e49-e658-4254-acb0-852486da7cd6&projectCode=2106-440100-04-01-237448-2001&timestamp=20240303000000000"
# SM3 加密结果: c2e1be8596f77123c221d7a3cb2bf8b9a60344c2df4aa7031b0a997440725481
key = "18fb264463a05a42"
# 对数据进行SM3哈希


#data = "accessNo=e85c946c15f0338a473a9e75dfc6e17e&projectCode=4e9d4e49-e658-4254-acb0-852486da7cd6&factoryNum=4e9d4e49-e658-4254-acb0-852486da7cd6&clientId=87a72000-e517-4997-8bdb-62d248c9021a&timestamp=1712024721000&realNameSystemState=0&staffId=SM4:aa59fb73b32b90f87bad2cedc3b80fc8f1fbf21596e45df1dccd268d51d0f12f&exceptionTyp=0&accessDirection=1&accessTime=1712024721000&pushTime=1712024721000"
# SM3 加密结果: bcdbd75a49e6b064062d7a0b911b92a1bb3b4ab7cc64d69cc6c52f03b7fdc519
# 测试

#data = "accessNo=e85c946c15f0338a473a9e75dfc6e17e&projectCode=2106-440100-04-01-237448-2001&factoryNum=4e9d4e49-e658-4254-acb0-852486da7cd6&clientId=87a72000-e517-4997-8bdb-62d248c9021a&timestamp=1712024721000&realNameSystemState=0&staffId=SM4:aa59fb73b32b90f87bad2cedc3b80fc8f1fbf21596e45df1dccd268d51d0f12f&exceptionTyp=0&accessDirection=1&accessTime=1712024721000&pushTime=1712024721000"
# bcdbd75a49e6b064062d7a0b911b92a1bb3b4ab7cc64d69cc6c52f03b7fdc519

#data = "accessNo=e85c946c15f0338a473a9e75dfc6e17e&projectCode=2106-440100-04-01-237448-2001&factoryNum=4e9d4e49-e658-4254-acb0-852486da7cd6&clientId=87a72000-e517-4997-8bdb-62d248c9021a&timestamp=&realNameSystemState=&staffId=&exceptionTyp=&accessDirection=&accessTime=&pushTime="
# 9517b2aa7884d965f4dd6b4783fa400a51114df95deccfb009682b4c04f1ac1c


# data = "accessNo=e85c946c15f0338a473a9e75dfc6e17e&projectCode=2106-440100-04-01-237448-2001&factoryNum=4e9d4e49-e658-4254-acb0-852486da7cd6&clientId=87a72000-e517-4997-8bdb-62d248c9021a&timestamp=20230425150750729&anayseStarDate=2023-04-25+15%3A07%3A00&anayseEndDate=2023-04-25+15%3A07%3A00&estimatedQuantity=20&whiteQuantity=15&strangerQuantity=5&enterNum=2&outNum=2&pushTime=2023-04-25+15%3A07%3A0018fb264463a05a42"
# a313d54174a5160fd2252709f2bd621a88aeb5b9d8d9831bfe39c154d49f401d

# 香雪小学 - 人脸底库同步
data = "accessNo=332202a5ee8c95421a36b840030635bd&factoryNum=4e9d4e49-e658-4254-acb0-852486da7cd6&projectCode=2205-440116-04-01-895542-2001&timestamp=20240303000000000"
key = "6878f19ecb683da9"
# 166
# aa4543c7f367e8710b545c7e80a12c8f728e7e09ca68aa94b6a936a61aee95e2

# 166
# c81f1845a3b26d4e40199f9eb3853f8cd4560ebaf9e4a647e242f0610d8ecdc5

data="accessNo=332202a5ee8c95421a36b840030635bd&projectCode=2205-440116-04-01-895542-2001&factoryNum=4e9d4e49-e658-4254-acb0-852486da7cd6&clientId=088120cc-876d-475c-8217-ff0698a901ea&realNameSystemState=1&idCardNo=SM4%3Add85dd84bce5bab8aac280f7c65b16d60f763f74d6e9ff86c9bdd12f5420b1fd&userType=0&accessDirection=2&accessTime=2024-05-15+11%3A05%3A16&timestamp=202405151105285836878f19ecb683da9"

encrypted_message = sm3_encrypt(data+key)
print("SM3 加密结果:", encrypted_message.lower())