# worker.py
import redis
from db import get_db_connection
import json
from config import config
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from deepface import DeepFace
import uuid
import os
import hashlib

# Initialize Redis
redis_client = redis.StrictRedis.from_url(config.REDIS_URL)

# Define the backend and alignment mode
DETECTOR_BACKEND = 'retinaface'  # Choose from the provided backends
ALIGN_MODE = True  # Whether to align faces


def calculate_md5(image_data):
    """Calculate the MD5 hash of the image data."""
    md5 = hashlib.md5()
    md5.update(image_data)
    return md5.hexdigest()


def process_image(image_data, image_key, origin, image_url=None):
    # Calculate the MD5 hash of the image data
    if isinstance(image_data, str):
        image_data = base64.b64decode(image_data)
    image_md5 = calculate_md5(image_data)

    conn = get_db_connection()
    cur = conn.cursor()

    # Check if the image hash already exists in the database
    cur.execute("SELECT id FROM images WHERE md5_hash = %s;", (image_md5,))
    existing_image = cur.fetchone()

    if existing_image:
        print(f"Image already processed with ID: {existing_image[0]}. Skipping processing.")
        redis_client.delete(image_key)  # Clean up Redis
        cur.close()
        conn.close()
        return  # Exit the function to avoid processing the same image again

    # Proceed with image processing since the image is new
    img = Image.open(BytesIO(image_data))

    # Convert the image to a numpy array (BGR format)
    img_np = np.array(img)

    # Analyze the image to detect faces and extract embeddings
    try:
        embeddings = DeepFace.represent(
            img_path=img_np,  # Use numpy array directly
            model_name="Facenet512",
            detector_backend=DETECTOR_BACKEND,
            align=ALIGN_MODE,
            enforce_detection=False  # Do not raise an error if no faces are detected
        )
    except ValueError as e:
        print(f"No faces detected: {e}")
        redis_client.delete(image_key)  # Clean up Redis
        cur.close()
        conn.close()
        return  # Exit the function without saving or creating any database entry

    # If no embeddings are found, skip saving the image and making any database entry
    if not embeddings:
        print("No faces detected in the image.")
        redis_client.delete(image_key)  # Clean up Redis
        cur.close()
        conn.close()
        return  # Exit the function without saving or creating any database entry

    # Analyze the image for additional attributes
    face_data = DeepFace.analyze(
        img_path=img_np,  # Use numpy array directly
        actions=['age', 'gender', 'race', 'emotion'],
        detector_backend=DETECTOR_BACKEND,
        align=ALIGN_MODE,
        enforce_detection=False  # Do not raise an error if no faces are detected
    )

    # Generate a unique filename for the original image (if needed for future reference)
    unique_filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join('uploads', unique_filename)

    # Save the original image to disk
    img.save(image_path)

    # Insert the original image metadata, including the MD5 hash, into the database and get the image ID
    cur.execute(
        "INSERT INTO images (image_path, origin, url, md5_hash) VALUES (%s, %s, %s, %s) RETURNING id;",
        (unique_filename, origin, image_url, image_md5)
    )
    image_id = cur.fetchone()[0]

    for i, face in enumerate(face_data):
        embedding = embeddings[i]['embedding']
        age = face['age']
        gender = face['gender']
        race = face['dominant_race']
        emotion = face['dominant_emotion']
        face_position = face['region']
        selected_gender = max(gender, key=gender.get)

        # Convert embedding to a format suitable for querying
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'

        # Check for an existing person with a similar embedding
        cur.execute("""
            SELECT person_id, embedding <-> %s AS distance
            FROM face_embeddings
            WHERE embedding <-> %s <= %s
            ORDER BY distance ASC LIMIT 1;
        """, (embedding_str, embedding_str, 23.56))  # Adjust threshold as needed

        result = cur.fetchone()

        # Calculate the bounding box coordinates from face_position
        x0 = face_position['x']
        y0 = face_position['y']
        x1 = x0 + face_position['w']
        y1 = y0 + face_position['h']

        if result is not None:
            person_id, distance = result
            print(person_id, distance)
        else:
            # If no existing person matches, create a new person entry
            cur.execute("INSERT INTO persons DEFAULT VALUES RETURNING id;")
            person_id = cur.fetchone()[0]
            distance = 0
            print("New person created with ID:", person_id)

            # Crop the face from the image
            cropped_face = img.crop((x0, y0, x1, y1))

            # Save the cropped face image to disk
            cropped_face_filename = f"{uuid.uuid4()}_face.jpg"
            cropped_face_path = os.path.join('uploads', 'faces', cropped_face_filename)
            os.makedirs(os.path.dirname(cropped_face_path), exist_ok=True)
            cropped_face.save(cropped_face_path)

            # Update the person's record with the path to the cropped face image
            cur.execute("UPDATE persons SET face_image_path = %s WHERE id = %s;", (cropped_face_filename, person_id))

        # Ensure to commit the transaction if necessary
        conn.commit()

        # Insert the face embedding with the associated person_id
        cur.execute("""
            INSERT INTO face_embeddings (image_id, person_id, embedding, age, gender, race, emotion, distance, face_position) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, box(point(%s,%s), point(%s,%s)));
        """, (
            image_id, person_id, embedding_str, age, selected_gender, race, emotion, distance,
            x0, y0, x1, y1
        ))

    conn.commit()
    cur.close()
    conn.close()

    # Clean up Redis after processing
    redis_client.delete(image_key)
    print(f"Processed image {image_id} and saved to {image_path}")


def worker():
    while True:
        message = redis_client.blpop('image_queue')[1]
        data = json.loads(message)

        image_key = data['image_key']
        origin = data['origin']
        image_url = data.get('image_url')

        # Retrieve the image data from Redis using the key
        image_data = redis_client.get(image_key)

        if image_data:
            # Process the image directly from the data in Redis
            process_image(image_data, image_key, origin, image_url)
        else:
            print(f"Failed to retrieve image data for key {image_key}")


if __name__ == "__main__":
    worker()
