from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'spottheplace/webapp/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    with app.app_context():
        from .routes import init_routes
        init_routes(app)

    return app
