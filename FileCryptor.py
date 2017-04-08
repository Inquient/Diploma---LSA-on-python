from Crypto.Cipher import AES
from Crypto import Random
import hashlib


class FileCryptor:

    def __init__(self, password, filepath):
        self.key = hashlib.sha256(password).digest()        # Генерируем из пароля 32-битный ключ с помощью SHA-256
        self.filepath = filepath
        self.IV = Random.new().read(AES.block_size)         # Создаём рандомный инициализирующий вектор

    def pad(self, text):
        """Сообщение, которое необходимо зашифровать можеть иметь длинну НЕ кратную 16.
        Поскольку AES может шифровать только такие объекты, нужно просто дополнить текст
        количеством пробелов, равным разности между длинной текста и ближайшим числом, кратным 16."""
        return text + " " * (AES.block_size - len(text) % AES.block_size)

    def encrypt(self, text):
        text = self.pad(text)

        mode = AES.MODE_CBC                                 # Выбираем вид шифрования
        encryptor = AES.new(self.key, mode, IV=self.IV)     # Объявляем свой "шифровальщик"

        return self.IV+encryptor.encrypt(text)

    def decrypt(self, ciphertext):

        self.IV = ciphertext[:AES.block_size]
        mode = AES.MODE_CBC
        decryptor = AES.new(self.key, mode, IV=self.IV)     # Объявляем "дешифровальщик"

        return decryptor.decrypt(ciphertext[AES.block_size:])

    def encrypt_file(self, text):
        with open(self.filepath, 'wb') as f:
            f.write(self.encrypt(text))

    def decrypt_file(self):
        with open(self.filepath, 'rb') as f:
            ciphertext = f.read()
        return self.decrypt(ciphertext)
