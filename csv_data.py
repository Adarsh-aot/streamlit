import csv
import chromadb
from chromadb.utils import embedding_functions



with open('./output.csv' ,  encoding="utf-8") as file:
    lines = csv.reader(file)

    
    documents = []

   
    metadatas = []

    
    ids = []
    id = 1

    
    for i, line in enumerate(lines):
        if i == 0:
            
            continue

        documents.append(line[0])
        metadatas.append({"item_id": line[1]})
        ids.append(str(id))
        id += 1


chroma_client = chromadb.PersistentClient(path="data_db")


sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")


collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=sentence_transformer_ef)


collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)



    
            
