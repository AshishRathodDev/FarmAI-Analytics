import os
from flask import Flask, send_from_directory


static_folder = os.path.join(os.getcwd(), 'dist')

app = Flask(__name__, static_folder=static_folder, static_url_path='')

@app.route('/')
def serve():
    return send_from_directory(static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    file_path = os.path.join(static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(static_folder, path)
    # React Router ke liye fallback
    return send_from_directory(static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)