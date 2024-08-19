from flask import Flask, request, flash, redirect, url_for, render_template, send_from_directory
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
import base64

app = Flask(__name__)
app.config.from_object(config)

# Initialize Redis
redis_client = redis.StrictRedis.from_url(app.config['REDIS_URL'])

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tiff', 'tif'}
# Directory for uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/faces/<filename>')
def face_images(filename):
    return send_from_directory(f"{app.config['UPLOAD_FOLDER']}/faces", filename)

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
        image_data = None
        image_key = f"image:{uuid.uuid4()}"

        if 'image_file' in request.files and request.files['image_file'].filename != '':
            file = request.files['image_file']
            if allowed_file(file.filename):
                # Convert the uploaded file to JPEG in memory
                converted_image = convert_image(file, format='JPEG')
                image_data = converted_image.getvalue()  # Get the bytes data
                origin = 'local'
                flash('Image uploaded successfully from disk!', 'success')
            else:
                flash('Invalid file type. Please upload a valid image file.', 'error')
                return redirect(url_for('upload_image'))

        elif 'image_url' in request.form and request.form['image_url'] != '':
            image_url = request.form['image_url']
            response = requests.get(image_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                if img.format.lower() in ALLOWED_EXTENSIONS:
                    # Convert the image to JPEG in memory
                    converted_image = convert_image(BytesIO(response.content), format='JPEG')
                    image_data = converted_image.getvalue()  # Get the bytes data
                    origin = 'url'
                    flash('Image uploaded successfully from URL!', 'success')
                else:
                    flash('Invalid image format from URL. Please provide a valid image URL.', 'error')
                    return redirect(url_for('upload_image'))
            else:
                flash('Failed to download image from URL. Please check the URL and try again.', 'error')
                return redirect(url_for('upload_image'))

        elif 'image_base64' in request.form and request.form['image_base64'] != '':
            image_base64 = request.form['image_base64']

            try:
                # Decode the base64 string and convert it to an image
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data))

                if img.format.lower() in ALLOWED_EXTENSIONS:
                    # Convert the image to JPEG in memory
                    converted_image = convert_image(BytesIO(image_data), format='JPEG')
                    image_data = converted_image.getvalue()  # Get the bytes data
                    origin = 'base64'
                    flash('Image uploaded successfully from base64 string!', 'success')
                else:
                    flash('Invalid image format from base64 string. Please provide a valid image.', 'error')
                    return redirect(url_for('upload_image'))

            except (base64.binascii.Error, IOError) as e:
                flash(f'Failed to decode and process base64 image: {str(e)}', 'error')
                return redirect(url_for('upload_image'))

        else:
            flash('No image provided. Please upload an image, provide a URL, or a base64 string.', 'error')
            return redirect(url_for('upload_image'))

        # Save the image data to Redis
        if image_data:
            redis_client.set(image_key, image_data)
            redis_client.rpush('image_queue', json.dumps({
                'image_key': image_key,
                'origin': origin,
                'image_url': request.form.get('image_url', None)  # Add URL if available
            }))
            flash(f"Image successfully uploaded and queued for processing (key: {image_key})", 'success')

        return redirect(url_for('upload_image'))

    return render_template('upload.html')



# Route for displaying the list of persons
@app.route('/persons', methods=['GET'])
def list_persons():
    conn = get_db_connection()
    cur = conn.cursor()

    # Get all persons with their face image paths
    cur.execute("SELECT id, face_image_path FROM persons ORDER BY id ASC;")
    persons = cur.fetchall()

    cur.close()
    conn.close()

    return render_template('persons.html', persons=persons)


# Route for displaying all images associated with a person
@app.route('/person/<int:person_id>', methods=['GET'])
def show_person(person_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # Get all images and the cropped face image path where this person appears
    cur.execute("""
        SELECT i.image_path, f.face_position, p.face_image_path
        FROM images i 
        JOIN face_embeddings f ON i.id = f.image_id 
        JOIN persons p ON f.person_id = p.id
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
        SELECT f.person_id, f.age, f.gender, f.race, f.emotion, f.face_position, f.distance, i.origin, i.url
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

    bbox = face[5]  # face_position, now directly use the tuple of coordinates
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

    # Prepare data for rendering the template
    image_info = {
        "image_path": f"processed_{os.path.basename(image_path)}",
        "person_id": face[0],
        "age": face[1],
        "gender": face[2],
        "race": face[3],
        "emotion": face[4],
        "bbox": face[5],
        "distance": face[6],  # Add the distance value here
        "origin": face[7],
        "url": face[8],
    }

    # Render the template with the image, face metadata, and additional info
    return render_template('image.html', image_info=image_info)


# Route for displaying clusters
@app.route('/clusters', methods=['GET'])
def list_clusters():
    conn = get_db_connection()
    cur = conn.cursor()

    # Get all clusters and their associated faces
    cur.execute("""
        SELECT f.cluster_id, p.id as person_id, p.face_image_path, i.image_path 
        FROM face_embeddings f 
        JOIN persons p ON f.person_id = p.id
        JOIN images i ON f.image_id = i.id
        WHERE f.cluster_id IS NOT NULL 
        ORDER BY f.cluster_id, p.id ASC;
    """)
    clusters = cur.fetchall()

    cur.close()
    conn.close()

    # Group faces by cluster ID
    clusters_dict = {}
    for cluster_id, person_id, face_image_path, image_path in clusters:
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = []
        clusters_dict[cluster_id].append({
            'person_id': person_id,
            'face_image_path': face_image_path,
            'image_path': image_path
        })

    return render_template('clusters.html', clusters=clusters_dict)



if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=8080)
