from flask import Flask, request, jsonify, render_template, send_from_directory
import redis
import uuid
import os
import requests
from io import BytesIO
from db import get_db_connection, init_db
import json
from config import config
from PIL import Image, ImageDraw
import ast

app = Flask(__name__)
app.config.from_object(config)

# Initialize Redis
redis_client = redis.StrictRedis.from_url(app.config['REDIS_URL'])

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tiff'}
# Directory for uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def uploaded_file(filename):
    return s@app.route('/uploads/<filename>')
end_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_image(image, format='JPEG'):
    """Convert image to a specific format (default: JPEG)."""
    img = Image.open(image)
    img = img.convert('RGB')  # Convert to RGB (required for JPEG)
    output = BytesIO()
    img.save(output, format=format)
    output.seek(0)
    return output

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        origin = ''
        image_url = None

        if 'image_file' in request.files and request.files['image_file'].filename != '':
            file = request.files['image_file']
            if allowed_file(file.filename):
                unique_filename = f"{uuid.uuid4()}.jpg"
                image_path = os.path.join('uploads', unique_filename)

                # Convert and save the image as JPEG
                converted_image = convert_image(file, format='JPEG')
                with open(image_path, 'wb') as f:
                    f.write(converted_image.getbuffer())

                origin = 'local'
            else:
                return jsonify({"status": "error", "message": "Invalid file type"}), 400

        elif 'image_url' in request.form and request.form['image_url'] != '':
            image_url = request.form['image_url']
            response = requests.get(image_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                if img.format.lower() in ALLOWED_EXTENSIONS:
                    unique_filename = f"{uuid.uuid4()}.jpg"
                    image_path = os.path.join('uploads', unique_filename)

                    # Convert and save the image as JPEG
                    converted_image = convert_image(BytesIO(response.content), format='JPEG')
                    with open(image_path, 'wb') as f:
                        f.write(converted_image.getbuffer())

                    origin = 'url'
                else:
                    return jsonify({"status": "error", "message": "Invalid image format from URL"}), 400
            else:
                return jsonify({"status": "error", "message": "Failed to download image"}), 400
        else:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        # Store the image path, origin, and URL (if applicable) in the database

        filename = image_path.split('/')[-1]

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO images (image_path, origin, url) VALUES (%s, %s, %s) RETURNING id;",
            (filename, origin, image_url)
        )
        image_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        # Add the image path to the Redis queue for processing
        redis_client.rpush('image_queue', json.dumps({'image_path': image_path, 'image_id': image_id}))

        return jsonify({"status": "success", "image_id": image_id})

    return render_template('upload.html')



# Route for displaying the list of persons
@app.route('/persons', methods=['GET'])
def list_persons():
    conn = get_db_connection()
    cur = conn.cursor()

    # Get all unique persons
    cur.execute("SELECT DISTINCT person_id FROM face_embeddings ORDER BY person_id ASC;")
    persons = cur.fetchall()

    cur.close()
    conn.close()

    return render_template('persons.html', persons=persons)

# Route for displaying all images associated with a person
@app.route('/person/<int:person_id>', methods=['GET'])
def show_person(person_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # Get all images where this person appears
    cur.execute("""
        SELECT i.image_path, f.face_position 
        FROM images i 
        JOIN face_embeddings f ON i.id = f.image_id 
        WHERE f.person_id = %s
        ORDER BY i.upload_time DESC;
    """, (person_id,))
    images = cur.fetchall()

    cur.close()
    conn.close()

    return render_template('person.html', person_id=person_id, images=images)

# Route for displaying details of a specific image and face
@app.route('/person/<int:person_id>/image/<path:image_path>', methods=['GET'])
def show_image(person_id, image_path):
    conn = get_db_connection()
    cur = conn.cursor()

    # Get the face and metadata in this image for the given person_id
    cur.execute("""
        SELECT f.person_id, f.age, f.gender, f.race, f.emotion, f.face_position 
        FROM images i 
        JOIN face_embeddings f ON i.id = f.image_id 
        WHERE i.image_path = %s AND f.person_id = %s;
    """, (image_path, person_id))
    face = cur.fetchone()  # Expecting only one face

    cur.close()
    conn.close()

    if face is None:
        return "No face found for this person in the given image.", 404

    # Draw bounding box on the image
    image_full_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
    image = Image.open(image_full_path)
    draw = ImageDraw.Draw(image)

    bbox = face[5]  # face_position
    if bbox:
        # Parse the bbox string
        bbox = ast.literal_eval(f"[{bbox}]")  # Converts the string to a list of tuples
        (x1, y1), (x2, y2) = bbox  # Extract the coordinates

        # Ensure correct order for drawing
        x0, x1 = sorted([x1, x2])
        y0, y1 = sorted([y1, y2])

        print(f"Drawing rectangle from ({x0}, {y0}) to ({x1}, {y1})")

        # Draw the rectangle
        draw.rectangle([x0, y0, x1, y1], outline="red", width=15)

    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{os.path.basename(image_path)}")
    image.save(processed_image_path)

    return render_template('image.html', image_path=f"processed_{os.path.basename(image_path)}", face=face)



if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=8080)
