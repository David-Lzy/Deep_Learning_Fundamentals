import base64
import hashlib
import os

from cryptography.fernet import Fernet
from binance.spot import Spot


def generate_key(password):
    """Generate a Fernet key from the given password."""
    password = password.encode("utf-8")
    key = hashlib.sha256(password).digest()
    return base64.urlsafe_b64encode(key)


def decrypt(encrypted_api_key, encrypted_api_secret, password):
    """Decrypt the given encrypted API key and encrypted API secret using the given password."""

    key = generate_key(password)
    f = Fernet(key)
    return (
        f.decrypt(encrypted_api_key).decode("utf-8"),
        f.decrypt(encrypted_api_secret).decode("utf-8"),
    )


def encrypt(api_key, api_secret, password):
    """Encrypt the given API key and API secret using the given password."""
    key = generate_key(password)
    f = Fernet(key)

    encrypted_api_key = f.encrypt(api_key.encode("utf-8"))
    encrypted_api_secret = f.encrypt(api_secret.encode("utf-8"))

    return encrypted_api_key, encrypted_api_secret


Encrypted_API_key = b"gAAAAABkBftUAQpiQIq_tiao3Oo_1Ide4UxzqFFeBmPPeJck9fXzh1dzhaMESPYSgtT0n0pl7GU3SbDYwLE6QvDTC3c5hkYkbNfNdMlLFfiududUilu7QOmympGhAR-t0Nb9v5YoEFgRELMEP0M-L4gd6uJuRZYEsIfcJ9yEV4Ni-c47eMvkrA0="

Encrypted_API_secret = b"gAAAAABkBftUuirQojixItFoqGEhEt_yuwesE4d72YGbD87PoMZbcW7WLzT_ojrMObtpvOT0EQP8rZnmEDNjWD8BJCT73u5OKFuuGZHqJ2AOU_mIZJzhweMHyqyBzEwmzABIjbTGqik2Kk137FEw-MpXKoryGQzVCLJd1vRKYG8hqEJKyjY8BRc="

if __name__ == "__main__":
    api_key = "test1"
    api_secret = "test2"
    password = "test3"

    encrypted_api_key, encrypted_api_secret = encrypt(api_key, api_secret, password)

    print(f"Encrypted API key: {encrypted_api_key}")
    print(f"Encrypted API secret: {encrypted_api_secret}")

    decrypted_api_key, decrypted_api_secret = decrypt(
        encrypted_api_key, encrypted_api_secret, password
    )

    print(f"Decrypted API key: {decrypted_api_key}")
    print(f"Decrypted API secret: {decrypted_api_secret}")
