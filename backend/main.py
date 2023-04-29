from flask import Flask, request

app = Flask(__name__)


@app.route("/model/<model_name>")
def hello_world(model_name):
    user_id = request.args.get('user_id')
    return f'Hello, World! {model_name} {user_id}'
