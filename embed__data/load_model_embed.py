from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
                                         model_kwargs={"device": "cpu"},
                                         encode_kwargs = {'normalize_embeddings':True}
                                         )