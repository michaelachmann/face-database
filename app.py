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

def add_base64_padding(base64_string):
    return base64_string + '=' * (-len(base64_string) % 4)


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
                origin = 'file'
                flash('Image uploaded successfully from disk!', 'success')
            else:
                flash('Invalid file type. Please upload a valid image file.', 'error')
                return redirect(url_for('upload_image'))


        elif 'image_base64' in request.form and request.form['image_base64'] != '':
            image_base64 = request.form['image_base64']

            try:
                # Fix the padding if necessary
                image_base64 = add_base64_padding(image_base64)

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

    # Get filter parameters
    min_images = request.args.get('min_images', default=0, type=int)
    max_age = request.args.get('max_age', default=None, type=float)
    name_filter = request.args.get('name', default=None, type=str)

    # Get pagination and sorting parameters
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=12, type=int)
    sort_by = request.args.get('sort_by', default='id', type=str)
    sort_order = request.args.get('sort_order', default='asc', type=str)

    # Start building the SQL query
    query = '''
        SELECT p.id, p.face_image_path, p.name, COUNT(fe.image_id) as image_count, 
               AVG(fe.age) as mean_age
        FROM persons p
        LEFT JOIN face_embeddings fe ON p.id = fe.person_id
    '''

    # Initialize the WHERE clause
    conditions = []
    if name_filter:
        conditions.append('p.name ILIKE %s')

    # Add the conditions to the query
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)

    # Add the GROUP BY clause
    query += ' GROUP BY p.id, p.face_image_path, p.name'

    # Add the HAVING clause for aggregate filters
    having_conditions = []
    if min_images > 0:
        having_conditions.append('COUNT(fe.image_id) >= %s')
    if max_age is not None:
        having_conditions.append('AVG(fe.age) <= %s')

    if having_conditions:
        query += ' HAVING ' + ' AND '.join(having_conditions)

    # Add the ORDER BY clause
    query += f' ORDER BY {sort_by} {sort_order}'

    # Add LIMIT and OFFSET for pagination
    query += ' LIMIT %s OFFSET %s'

    # Prepare the parameters for the query
    params = []
    if name_filter:
        params.append(f'%{name_filter}%')
    if min_images > 0:
        params.append(min_images)
    if max_age is not None:
        params.append(max_age)

    params.extend([per_page, (page - 1) * per_page])

    cur.execute(query, tuple(params))
    persons = cur.fetchall()

    # Get the total number of records for pagination
    cur.execute('''
        SELECT COUNT(DISTINCT p.id)
        FROM persons p
        LEFT JOIN face_embeddings fe ON p.id = fe.person_id
    ''')
    total_persons = cur.fetchone()[0]

    cur.close()
    conn.close()

    # Calculate total pages
    total_pages = (total_persons + per_page - 1) // per_page

    return render_template('persons.html', persons=persons, page=page, per_page=per_page, total_pages=total_pages,
                           sort_by=sort_by, sort_order=sort_order)


@app.route('/persons/<int:person_id>/edit', methods=['GET', 'POST'])
def edit_person(person_id):
    conn = get_db_connection()
    cur = conn.cursor()

    if request.method == 'POST':
        name = request.form['name']
        cur.execute('''
            UPDATE persons 
            SET name = %s 
            WHERE id = %s
        ''', (name, person_id))
        conn.commit()
        cur.close()
        conn.close()
        return redirect(url_for('list_persons'))

    cur.execute('''
        SELECT id, name 
        FROM persons 
        WHERE id = %s
    ''', (person_id,))
    person = cur.fetchone()

    cur.close()
    conn.close()

    return render_template('edit_person.html', person=person)


# Route for displaying all images associated with a person
@app.route('/person/<int:person_id>', methods=['GET'])
def show_person(person_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # Get person details including name, mean age, and image count
    cur.execute('''
        SELECT p.name, AVG(fe.age) as mean_age, COUNT(fe.image_id) as image_count
        FROM persons p
        LEFT JOIN face_embeddings fe ON p.id = fe.person_id
        WHERE p.id = %s
        GROUP BY p.id;
    ''', (person_id,))
    person_details = cur.fetchone()

    # Get all images and metadata where this person appears
    cur.execute('''
        SELECT i.image_path, f.face_position, p.face_image_path, i.upload_time, i.origin, i.url, f.id
        FROM images i 
        JOIN face_embeddings f ON i.id = f.image_id 
        JOIN persons p ON f.person_id = p.id
        WHERE f.person_id = %s
        ORDER BY i.upload_time DESC;
    ''', (person_id,))
    images = cur.fetchall()

    cur.close()
    conn.close()

    return render_template('person.html', person_id=person_id, person_details=person_details, images=images)


@app.route('/person/<int:person_id>/image/<path:image_path>', methods=['GET'])
def show_image(person_id, image_path):
    conn = get_db_connection()
    cur = conn.cursor()

    # Get the face and metadata in this image for the given person_id
    cur.execute("""
        SELECT f.person_id, f.age, f.gender, f.race, f.emotion, f.face_position, f.distance, i.origin, i.url, f.embedding
        FROM images i 
        JOIN face_embeddings f ON i.id = f.image_id 
        WHERE i.image_path = %s AND f.person_id = %s;
    """, (image_path, person_id))
    face = cur.fetchone()  # Expecting only one face

    if face is None:
        cur.close()
        conn.close()
        return "No face found for this person in the given image.", 404

    embedding = face[9]  # The embedding vector

    cur.execute("""
        SELECT i.image_path, f.person_id, (f.embedding <-> %s) AS calculated_distance
        FROM face_embeddings f
        JOIN images i ON f.image_id = i.id
        WHERE i.image_path != %s
        ORDER BY calculated_distance ASC
        LIMIT 4;
    """, (embedding, image_path))

    similar_faces = cur.fetchall()
    cur.close()
    conn.close()

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
        draw.rectangle([x0, y0, x1, y1], outline="red", width=5)

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
        "distance": face[6],
        "origin": face[7],
        "url": face[8],
        "image_uuid": image_path
    }

    # Prepare the list of similar faces
    similar_faces_data = [{
        "image_path": f[0],
        "person_id": f[1],
        "distance": f[2]  # This is the calculated distance
    } for f in similar_faces]

    # Render the template with the image, face metadata, and similar faces
    return render_template('image.html', image_info=image_info, similar_faces=similar_faces_data)



# Route for displaying clusters
@app.route('/clusters', methods=['GET'])
def list_clusters():
    conn = get_db_connection()
    cur = conn.cursor()

    # Query to fetch all images, grouped by cluster, and retrieve the age connected to each image
    cur.execute("""
        SELECT f.cluster_id, p.id as person_id, p.name, p.face_image_path, 
               i.image_path, i.upload_time, i.origin, i.url,
               f.age, COUNT(fe2.image_id) as image_count
        FROM face_embeddings f 
        JOIN persons p ON f.person_id = p.id
        JOIN images i ON f.image_id = i.id
        LEFT JOIN face_embeddings fe2 ON p.id = fe2.person_id
        WHERE f.cluster_id IS NOT NULL
        GROUP BY f.cluster_id, p.id, p.face_image_path, i.image_path, i.upload_time, i.origin, i.url, f.age
        ORDER BY f.cluster_id, p.id ASC
    """)
    clusters = cur.fetchall()

    cur.close()
    conn.close()

    # Group faces by cluster ID, limit to 5 images per cluster in overview
    clusters_dict = {}
    cluster_image_counts = {}  # To store the number of images per cluster

    for cluster_id, person_id, name, face_image_path, image_path, upload_time, origin, url, age, image_count in clusters:
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = []
            cluster_image_counts[cluster_id] = 0  # Initialize count

        if len(clusters_dict[cluster_id]) < 4:
            clusters_dict[cluster_id].append({
                'person_id': person_id,
                'name': name,
                'face_image_path': face_image_path,
                'image_path': image_path,
                'upload_time': upload_time,
                'origin': origin,
                'url': url,
                'age': age,
                'image_count': image_count
            })

        cluster_image_counts[cluster_id] += 1  # Increment the count

    return render_template('clusters_overview.html', clusters=clusters_dict, cluster_image_counts=cluster_image_counts)


@app.route('/clusters/<int:cluster_id>', methods=['GET'])
def show_cluster(cluster_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # Query to fetch all images in a specific cluster
    cur.execute("""
        SELECT f.cluster_id, p.id as person_id, p.name, p.face_image_path, 
               i.image_path, i.upload_time, i.origin, i.url,
               COUNT(fe2.image_id) as image_count, AVG(fe2.age) as mean_age
        FROM face_embeddings f 
        JOIN persons p ON f.person_id = p.id
        JOIN images i ON f.image_id = i.id
        LEFT JOIN face_embeddings fe2 ON p.id = fe2.person_id
        WHERE f.cluster_id = %s
        GROUP BY f.cluster_id, p.id, p.face_image_path, i.image_path, i.upload_time, i.origin, i.url
        ORDER BY p.id ASC
    """, (cluster_id,))
    images = cur.fetchall()

    cur.close()
    conn.close()

    # Group faces by cluster ID
    cluster_data = {
        'cluster_id': cluster_id,
        'images': []
    }

    for cluster_id, person_id, name, face_image_path, image_path, upload_time, origin, url, image_count, mean_age in images:
        cluster_data['images'].append({
            'person_id': person_id,
            'name': name,
            'face_image_path': face_image_path,
            'image_path': image_path,
            'upload_time': upload_time,
            'origin': origin,
            'url': url,
            'image_count': image_count,
            'mean_age': mean_age
        })

    return render_template('cluster_details.html', cluster=cluster_data)

@app.route('/analysis/overlap', methods=['GET'])
def analyze_overlap():
    conn = get_db_connection()
    cur = conn.cursor()

    # Query to fetch one sample image per cluster and one sample image per person within that cluster
    cur.execute("""
        SELECT 
            f.person_id, 
            f.cluster_id, 
            MIN(i.image_path) as person_image, 
            (SELECT MIN(i2.image_path) 
             FROM face_embeddings f2 
             JOIN images i2 ON f2.image_id = i2.id 
             WHERE f2.cluster_id = f.cluster_id 
             AND f2.cluster_id IS NOT NULL AND f2.cluster_id != -1) as cluster_image, 
            COUNT(*) as count
        FROM face_embeddings f
        JOIN images i ON f.image_id = i.id
        WHERE f.cluster_id IS NOT NULL AND f.cluster_id != -1
        GROUP BY f.person_id, f.cluster_id
        ORDER BY count DESC;
    """)

    overlap_data = cur.fetchall()

    cur.close()
    conn.close()

    # Structure the data for analysis
    overlap_dict = {}
    for person_id, cluster_id, person_image, cluster_image, count in overlap_data:
        if person_id not in overlap_dict:
            overlap_dict[person_id] = {}
        overlap_dict[person_id][cluster_id] = {
            'count': count,
            'person_image': person_image,
            'cluster_image': cluster_image
        }

    return render_template('analysis_overlap.html', overlap_dict=overlap_dict)


@app.route('/images', methods=['GET'])
def list_images():
    conn = get_db_connection()
    cur = conn.cursor()

    race_categories = [
        "asian",
        "black",
        "indian",
        "latino hispanic",
        "middle eastern",
        "white"
    ]

    # Get filter, sorting, and pagination parameters from the request
    person_id = request.args.get('person_id', default=None, type=int)
    gender = request.args.get('gender', default=None, type=str)
    race = request.args.get('race', default=None, type=str)
    age = request.args.get('age', default=None, type=int)
    cluster_id = request.args.get('cluster_id', default=None, type=int)
    sort_by = request.args.get('sort_by', default='upload_time', type=str)
    sort_order = request.args.get('sort_order', default='desc', type=str)
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=12, type=int)

    # Build the base query
    query = """
        SELECT i.image_path, f.person_id, f.age, f.gender, f.race, f.emotion, i.upload_time, f.distance, f.cluster_id, i.id
        FROM images i
        JOIN face_embeddings f ON i.id = f.image_id
        WHERE 1=1
    """

    # Add filters to the query
    params = []
    if person_id:
        query += " AND f.person_id = %s"
        params.append(person_id)
    if gender:
        query += " AND f.gender = %s"
        params.append(gender)
    if race:
        query += " AND f.race = %s"
        params.append(race)
    if age:
        query += " AND f.age = %s"
        params.append(age)
    if cluster_id:
        query += " AND f.cluster_id = %s"
        params.append(cluster_id)

    # Add sorting to the query
    query += f" ORDER BY {sort_by} {sort_order}"

    # Add pagination to the query
    offset = (page - 1) * per_page
    query += " LIMIT %s OFFSET %s"
    params.extend([per_page, offset])

    cur.execute(query, tuple(params))
    images = cur.fetchall()

    # Get the total number of images for pagination
    cur.execute("""
        SELECT COUNT(*)
        FROM images i
        JOIN face_embeddings f ON i.id = f.image_id
        WHERE 1=1
    """ + (" AND f.person_id = %s" if person_id else "") + \
         (" AND f.gender = %s" if gender else "") + \
         (" AND f.race = %s" if race else "") + \
         (" AND f.age = %s" if age else "") + \
         (" AND f.cluster_id = %s" if cluster_id else ""),
        tuple(params[:len(params) - 2])  # Exclude LIMIT and OFFSET params
    )
    total_images = cur.fetchone()[0]

    cur.close()
    conn.close()

    # Calculate total pages
    total_pages = (total_images + per_page - 1) // per_page

    return render_template('images.html',
                           images=images,
                           page=page,
                           per_page=per_page,
                           total_pages=total_pages,
                           sort_by=sort_by,
                           sort_order=sort_order,
                           max=max,
                           min=min,
                           race_categories=race_categories)

@app.route('/image/<int:image_id>/delete', methods=['POST'])
def delete_image(image_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve the image path
    cur.execute('SELECT image_path FROM images WHERE id = %s', (image_id,))
    image_record = cur.fetchone()
    if not image_record:
        flash('Image not found.', 'error')
        return redirect(url_for('list_images'))

    image_path = image_record[0]

    # Delete the image file from the filesystem
    try:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image_path))
    except FileNotFoundError:
        flash('Image file not found on the server.', 'warning')

    # Delete face embeddings associated with this image
    cur.execute('DELETE FROM face_embeddings WHERE image_id = %s', (image_id,))

    # Delete the image record from the database
    cur.execute('DELETE FROM images WHERE id = %s', (image_id,))

    conn.commit()
    cur.close()
    conn.close()

    flash('Image and associated embeddings deleted successfully.', 'success')
    return redirect(url_for('list_images'))

@app.route('/persons/merge', methods=['POST'])
def merge_persons():
    person_ids = request.form.getlist('person_ids')  # List of person IDs to merge
    target_person_id = request.form['target_person_id']  # ID of the person to keep

    if not person_ids or not target_person_id:
        flash('Please select persons to merge and a target person.', 'error')
        return redirect(url_for('list_persons'))

    conn = get_db_connection()
    cur = conn.cursor()

    # Convert person_ids to a list of integers
    person_ids = list(map(int, person_ids))

    # Update the face_embeddings to point to the target person
    cur.execute('''
        UPDATE face_embeddings
        SET person_id = %s
        WHERE person_id = ANY(%s) AND person_id != %s
    ''', (target_person_id, person_ids, target_person_id))

    # Optionally, delete the merged persons from the persons table
    cur.execute('''
        DELETE FROM persons WHERE id = ANY(%s) AND id != %s
    ''', (person_ids, target_person_id))

    conn.commit()
    cur.close()
    conn.close()

    flash('Persons merged successfully.', 'success')
    return redirect(url_for('list_persons'))



@app.route('/face_embedding/<int:embedding_id>/reassign', methods=['GET'])
def reassign_face_embedding_view(embedding_id):
    # Fetch all persons to populate the dropdown
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('''
        SELECT id, name
        FROM persons
        ORDER BY name ASC;
    ''')
    persons = cur.fetchall()

    cur.close()
    conn.close()

    return render_template('reassign_face_embedding.html', embedding_id=embedding_id, persons=persons)

@app.route('/face_embedding/<int:embedding_id>/reassign', methods=['POST'])
def reassign_face_embedding(embedding_id):
    new_person_id = request.form.get('new_person_id')
    create_new_person = request.form.get('create_new_person', False)

    conn = get_db_connection()
    cur = conn.cursor()

    if create_new_person:
        # Create a new person and get the ID
        cur.execute("INSERT INTO persons DEFAULT VALUES RETURNING id;")
        new_person_id = cur.fetchone()[0]

    if new_person_id:
        # Update the face_embedding to point to the new person
        cur.execute('''
            UPDATE face_embeddings
            SET person_id = %s
            WHERE id = %s
        ''', (new_person_id, embedding_id))

        conn.commit()
        cur.close()
        conn.close()

        flash('Face embedding reassigned successfully.', 'success')
        return redirect(url_for('list_persons'))

    flash('Failed to reassign face embedding.', 'error')
    return redirect(url_for('show_person', person_id=new_person_id))



if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=8080)
