# db.py
import psycopg2
from config import config


def get_db_connection():
    conn = psycopg2.connect(config.SQLALCHEMY_DATABASE_URI)
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            image_path TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            origin VARCHAR(10) NOT NULL,
            url TEXT,
            md5_hash TEXT UNIQUE NOT NULL
        );
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS persons (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) DEFAULT NULL,
        face_image_path TEXT DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id SERIAL PRIMARY KEY,
        image_id INT REFERENCES images(id) ON DELETE CASCADE,
        person_id INT REFERENCES persons(id) ON DELETE CASCADE,
        embedding VECTOR(512),  -- Using pgvector type for storing embeddings
        age INT,
        gender TEXT,
        race TEXT,
        emotion TEXT,
        distance FLOAT,
        face_position BOX,
        cluster_id INT DEFAULT -1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ''')

    # Create GIST index on face_position
    cur.execute('''
    CREATE INDEX IF NOT EXISTS face_position_idx ON face_embeddings USING GIST (face_position);
    CREATE INDEX IF NOT EXISTS face_embedding_vector_idx ON face_embeddings USING ivfflat (embedding) WITH (lists = 100);
    ''')

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    init_db()
