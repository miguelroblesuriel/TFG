import psycopg2
import json
from Model_training.add_padding import add_padding

if __name__ == '__main__':
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host='172.25.128.1',
        port="5432"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT scan, filename, embedding FROM embeddings_2;")
    rows = cursor.fetchall()

    print(f"Procesando {len(rows)} filas...")
    max_len = 0
    for scan, filename, emb_raw in rows:
        embedding = emb_raw.tobytes().decode('utf-8')
        embedding = json.loads(embedding)
        print(embedding)
        if len(embedding[0]) > max_len:
            max_len = len(embedding[0])
            print(max_len)

    for scan, filename, emb_raw in rows:
        embedding = emb_raw.tobytes().decode('utf-8')
        embedding = json.loads(embedding)
        print("Control de tamaño de embedding")
        print(len(embedding[0]))
        if len(embedding[0]) < max_len:
            embedding[0] = add_padding(embedding[0], max_len)
            embedding[1] = add_padding(embedding[1], max_len)
        print("COntrol de los embeddings de salida")
        print(len(embedding[0]), len(embedding[1]))
        embedding_bytes = json.dumps(embedding).encode('utf-8')
        cursor.execute(
            "UPDATE embeddings_2 SET embedding = %s WHERE scan = %s AND filename = %s",
            (embedding_bytes, scan, filename)
        )
    conn.commit()
    print("¡Base de datos actualizada con éxito!")
    cursor.close()
    conn.close()


