# config.py
import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = f"postgresql://user:password@postgres:5432/face_recognition"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REDIS_URL = "redis://redis:6379/0"


config = Config()
