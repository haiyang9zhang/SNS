from flask import Flask
from api.routes import weather_bp, init_models
from src.config import SERVER_CONFIG


def create_app():
    app = Flask(__name__)


    app.register_blueprint(weather_bp)


    print("Initializing the model and data processor...")
    if init_models():
        print("Initialization is successful! The server is ready。")
    else:
        print("Warning: Model initialization failed and the service may not work properly")

    return app


if __name__ == '__main__':
    app = create_app()
    print(f"Starting the server... Listening port {SERVER_CONFIG['port']}")
    app.run(
        host=SERVER_CONFIG['host'],
        port=SERVER_CONFIG['port'],
        debug=SERVER_CONFIG['debug']
    )