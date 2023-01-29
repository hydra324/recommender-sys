from flask import Flask, request, jsonify
from database import get_random_k
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/api/videos')
def get_top_k(k=10):
    user_id = request.args.get('userId')
    # top_k = []
    # return jsonify(videoLinks=top_k,userId=user_id)
    count = get_random_k()
    random_indices = np.random.randint(0,count,size=(k,))
    random_indices = random_indices.tolist()
    return jsonify(videoLinks=random_indices,userId=user_id)

if __name__ == "__main__":
    app.run(debug=True)