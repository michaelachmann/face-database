# worker.py
import redis
import psycopg2
from db import get_db_connection
from deepface import DeepFace
import json
from config import config

# Initialize Redis
redis_client = redis.StrictRedis.from_url(config.REDIS_URL)

# Define the backend and alignment mode
DETECTOR_BACKEND = 'retinaface'  # Choose from the provided backends
ALIGN_MODE = True  # Whether to align faces


def process_image(image_path, image_id):
    # Extract embeddings using DeepFace.represent()
    embeddings = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet512",
        detector_backend=DETECTOR_BACKEND,
        align=ALIGN_MODE
    )

    # Analyze the image for additional attributes
    face_data = DeepFace.analyze(
        img_path=image_path,
        actions=['age', 'gender', 'race', 'emotion'],
        detector_backend=DETECTOR_BACKEND,
        align=ALIGN_MODE
    )

    conn = get_db_connection()
    cur = conn.cursor()

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

        # Start a new transaction or ensure the previous one is committed/rolled back
        cur.execute("""
            SELECT person_id, embedding <-> %s AS distance
            FROM face_embeddings
            WHERE embedding <-> %s <= %s
            ORDER BY distance ASC LIMIT 1;
        """, (embedding_str, embedding_str, 23.56))  # Adjust threshold as needed

        result = cur.fetchone()

        if result is not None:
            person_id, distance = result
            print(person_id, distance)
        else:
            # If no existing person matches, create a new person entry
            cur.execute("INSERT INTO persons DEFAULT VALUES RETURNING id;")
            person_id = cur.fetchone()[0]
            distance = 0
            print("New person created with ID:", person_id)

        # Ensure to commit the transaction if necessary
        conn.commit()

        print(f"image_id: {image_id}, type: {type(image_id)}")
        print(f"person_id: {person_id}, type: {type(person_id)}")
        print(f"embedding_str: {embedding_str}, type: {type(embedding_str)}")
        print(f"age: {age}, type: {type(age)}")
        print(f"selected_gender: {selected_gender}, type: {type(selected_gender)}")
        print(f"race: {race}, type: {type(race)}")
        print(f"emotion: {emotion}, type: {type(emotion)}")
        print(f"distance: {distance}, type: {type(distance)}")
        print(f"face_position['x']: {face_position['x']}, type: {type(face_position['x'])}")
        print(f"face_position['y']: {face_position['y']}, type: {type(face_position['y'])}")
        print(
            f"face_position['x'] + face_position['w']: {face_position['x'] + face_position['w']}, type: {type(face_position['x'] + face_position['w'])}")
        print(
            f"face_position['y'] + face_position['h']: {face_position['y'] + face_position['h']}, type: {type(face_position['y'] + face_position['h'])}")

        # Extract the bounding box coordinates
        x0 = face_position['x']
        y0 = face_position['y']
        x1 = face_position['x'] + face_position['w']
        y1 = face_position['y'] + face_position['h']

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


def worker():
    while True:
        message = redis_client.blpop('image_queue')[1]
        data = json.loads(message)
        process_image(data['image_path'], data['image_id'])


if __name__ == "__main__":
    worker()
