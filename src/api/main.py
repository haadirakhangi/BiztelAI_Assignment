from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import logging
import os
import sys
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Adding src to Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data_preprocessing.loader import DataLoader
from src.data_preprocessing.cleaner import DataCleaner
from src.data_preprocessing.transformer import DataTransformer
from src.llm.insights import LLMInsightsExtractor 
from src.api import endpoints
from src.utils.helpers import clean_article_url_to_text

app_state = {
    "data_loader": None,
    "data_cleaner": None,
    "data_transformer": None,
    "df_processed": None,
    "sentence_model": None,
    "llm_insights_extractor": None,
    "faiss_index": None,
    "faiss_url_map": None,
    "raw_data_path": os.path.join(module_path, "data", "BiztelAI_DS_Dataset_V1.json"),
    "processed_data_path": os.path.join(module_path, "data", "processed_chat_data.pkl"),
    "fitted_transformer_path": os.path.join(module_path, "models", "fitted_data_transformer.pkl"),
    "faiss_index_path" : os.path.join(module_path, "models", "article_urls.index"),
    "faiss_url_map_path" : os.path.join(module_path, "models", "article_url_map.pkl"),
    "sentence_model_name": "all-MiniLM-L6-v2",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading resources...")
    try:
        os.makedirs(os.path.join(module_path, "models"), exist_ok=True)

        # Initialize core components
        app_state["data_loader"] = DataLoader(file_path=app_state["raw_data_path"])
        app_state["data_cleaner"] = DataCleaner()
        logger.info(f"Loading sentence transformer model: {app_state["sentence_model_name"]}...")
        app_state["sentence_model"] = SentenceTransformer(app_state["sentence_model_name"])
        logger.info("Sentence transformer model loaded.")
        
        # Handling DataTransformer, Processed DataFrame and FAISS index
        if os.path.exists(app_state["fitted_transformer_path"]) and os.path.exists(app_state["processed_data_path"]) and os.path.exists(app_state["faiss_index_path"]) and os.path.exists(app_state["faiss_url_map_path"]):

            # If all of the above exists, we will load it directly from the path
            logger.info(f"Loading fitted DataTransformer from {app_state['fitted_transformer_path']}")
            with open(app_state["fitted_transformer_path"], 'rb') as f:
                app_state["data_transformer"] = pickle.load(f)
            logger.info("Fitted DataTransformer loaded successfully.")

            logger.info(f"Loading Processed Dataset from {app_state['processed_data_path']}")
            app_state["df_processed"] = pd.read_pickle(app_state["processed_data_path"])
            logger.info("Processed Dataset loaded successfully.")

            logger.info("Loading FAISS index and URL map...")
            app_state["faiss_index"] = faiss.read_index(app_state["faiss_index_path"])
            with open(app_state["faiss_url_map_path"], 'rb') as f:
                app_state["faiss_url_map"] = pickle.load(f)
            logger.info(f"FAISS index loaded with {app_state["faiss_index"].ntotal} vectors.")

        else:
            # If any of the above does not exist, we will fit the data transformer, save it. Then create the processed dataframe and faiss index.
            logger.info("Fitted DataTransformer not found. Fitting a new one...")
            app_state["data_transformer"] = DataTransformer() 
            
            raw_dict = app_state["data_loader"].load_json_dataset()
            if raw_dict is None:
                raise RuntimeError("Failed to load raw data for fitting DataTransformer.")
            
            df_raw = app_state["data_loader"].structure_raw_dataset(raw_dict)
            df_cleaned = app_state["data_cleaner"].clean_data(df_raw.copy())
            
            app_state["df_processed"] = app_state["data_transformer"].transform_data(df_cleaned.copy()) 
            logger.info("DataTransformer fitted on initial dataset.")

            with open(app_state["fitted_transformer_path"], 'wb') as f:
                pickle.dump(app_state["data_transformer"], f)
            logger.info(f"New fitted DataTransformer saved to {app_state['fitted_transformer_path']}")

            app_state["df_processed"].to_pickle(app_state["processed_data_path"])
            logger.info(f"Generated and saved processed data to {app_state['processed_data_path']}. Shape: {app_state['df_processed'].shape}")
            unique_urls = df_raw['article_url'].unique().tolist()
            logger.info(f"Found {len(unique_urls)} unique URLs for FAISS index.")
            
            cleaned_url_texts = [clean_article_url_to_text(url) for url in unique_urls]
            valid_indices = [i for i, text in enumerate(cleaned_url_texts) if text.strip()]
            cleaned_url_texts_valid = [cleaned_url_texts[i] for i in valid_indices]
            unique_urls_valid = [unique_urls[i] for i in valid_indices]
            logger.info(f"Encoding {len(cleaned_url_texts_valid)} valid cleaned URL texts for FAISS...")
            url_embeddings = app_state["sentence_model"].encode(cleaned_url_texts_valid, convert_to_numpy=True)
            url_embeddings = url_embeddings.astype('float32')
            
            dimension = url_embeddings.shape[1]
            app_state["faiss_index"] = faiss.IndexFlatL2(dimension)
            app_state["faiss_index"].add(url_embeddings)
            
            # Map FAISS index (0 to N-1) to original URLs
            app_state["faiss_url_map"] = {i: original_url for i, original_url in enumerate(unique_urls_valid)}
            
            faiss.write_index(app_state["faiss_index"], app_state["faiss_index_path"])
            with open(app_state["faiss_url_map_path"], 'wb') as f:
                pickle.dump(app_state["faiss_url_map"], f)
            logger.info(f"FAISS index and URL map built and saved. Index size: {app_state["faiss_index"].ntotal}")

        if app_state["df_processed"] is None:
             logger.error("CRITICAL: df_processed is empty or None after startup. Summary endpoint will fail.")
        if app_state["data_transformer"] is None: 
            logger.error("CRITICAL: DataTransformer is not available or not fitted. Transformation endpoint will fail.")


        # Initializing LLM Extractor
        app_state["llm_insights_extractor"] = LLMInsightsExtractor() 
        logger.info("LLMInsightsExtractor initialized.")
        logger.info("Application startup complete.")

    except Exception as e:
        logger.error(f"Error during application startup: {e}", exc_info=True)

    app.state.app_state = app_state
    yield 

    logger.info("Application shutdown.")


app = FastAPI(lifespan=lifespan)

app.include_router(endpoints.router)

@app.get("/", tags=["Root"], include_in_schema=False) 
async def read_root():
    return {"message": "Welcome to the Biztel AI Internship API."}