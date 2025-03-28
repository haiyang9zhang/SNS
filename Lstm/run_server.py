import os
import sys


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)


from src.server import create_app, SERVER_CONFIG

if __name__ == '__main__':
    app = create_app()
    print(f"Starting the server... Listening on port {SERVER_CONFIG['port']}")
    app.run(
        host=SERVER_CONFIG['host'],
        port=SERVER_CONFIG['port'],
        debug=SERVER_CONFIG['debug']
    )