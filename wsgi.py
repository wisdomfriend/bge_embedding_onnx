"""
WSGI入口文件，用于生产环境部署（gunicorn）
"""
from app.main import app

application = app

