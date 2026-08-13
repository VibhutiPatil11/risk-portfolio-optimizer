import os
from dotenv import load_dotenv

load_dotenv()


def get_jwt_secret_key():
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret or len(secret.encode("utf-8")) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY must be set to a random value of at least 32 bytes."
        )
    return secret


class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:123456@localhost:5432/risk_portfolio_db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = get_jwt_secret_key()
