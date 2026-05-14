import psycopg2
import json


if __name__ == '__main__':
    conn = psycopg2.connect(
        dbname="embeddings",
        user="postgres",
        password="postgres",
        host='172.25.128.1',
        port="5432"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT scan, filename, embedding FROM embeddings;")
    rows = cursor.fetchall()

    for scan, filename, emb_raw in rows:
        embedding = emb_raw.tobytes().decode('utf-8')
        embedding = json.loads(embedding)
        masc = []
        for element in embedding[0]:
            if element:
                masc.append(0)
            else:
                masc.append(1)
        embedding.append(masc)
        embedding_bytes = json.dumps(embedding).encode('utf-8')
        cursor.execute(
            "UPDATE embeddings SET embedding = %s WHERE scan = %s AND filename = %s",
            (embedding_bytes, scan, filename)
        )
    conn.commit()
    print("¡Base de datos actualizada con éxito!")
    cursor.close()
    conn.close()

