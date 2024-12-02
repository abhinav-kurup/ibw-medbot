from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.vectorstores import Qdrant
import os

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
client = QdrantClient(path="./qdrant_storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
embed_dimension = len(embeddings.embed_query("test"))
try:
    client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME} already exists.")
except:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embed_dimension,
            distance=Distance.COSINE
        )
    )
    print(f"Created collection {COLLECTION_NAME}")

points = []
for i, chunk in enumerate(text_chunks):
    text = chunk.page_content
    metadata = chunk.metadata
    if not text:
        print(f"Skipping chunk {i} due to empty page_content")
        continue
    try:
        vector = embeddings.embed_query(text)
    except Exception as e:
        print(f"Error generating embedding for chunk {i}: {e}")
        continue

    point = PointStruct(
        id=i,  
        vector=vector,
        payload={
            'page_content': text, 
            **metadata 
        }
    )
    points.append(point)

if points:
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Successfully indexed {len(points)} documents in Qdrant.")
else:
    print("No documents to index. Please check your document loading process.")

print("Verifying indexed documents:")
points_info = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=5
)
for point in points_info[0]:
    print("Point Payload Keys:", point.payload.keys())
    print("Sample Page Content:", 
          str(point.payload.get('page_content', 'No content'))[:100])