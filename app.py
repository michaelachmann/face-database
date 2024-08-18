# app.py
from flask import Flask, request, jsonify
import redis
from db import get_db_connection, init_db
from deepface import DeepFace
import json
from config import config

app = Flask(__name__)
app.config.from_object(config)

# Initialize Redis
redis_client = redis.StrictRedis.from_url(app.config['REDIS_URL'])


@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image_path = f"uploads/{file.filename}"
    file.save(image_path)

    # Store the image path in the database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO images (image_path) VALUES (%s) RETURNING id;", (image_path,))
    image_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    # Add the image path to the Redis queue for processing
    redis_client.rpush('image_queue', json.dumps({'image_path': image_path, 'image_id': image_id}))

    return jsonify({"status": "success", "image_id": image_id})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=8080)
