from fastapi import APIRouter, HTTPException, Depends, Request
import pandas as pd
import logging

from src.api.schemas import (
    DatasetSummary, DatasetColumnSummary,
    RawMessageInputForTransform, ProcessedMessageOutput,
    TranscriptInput, InsightsOutput, Message
)
from src.llm.insights import LLMInsightsExtractor

logger = logging.getLogger(__name__)
router = APIRouter()

def get_app_state(request: Request) -> dict:
    return request.app.state.app_state

@router.get("/summary/dataset", response_model=DatasetSummary, tags=["Dataset Summary"])
async def get_dataset_summary(app_state: dict = Depends(get_app_state)):
    df = app_state["df_processed"]
    try:
        desc_raw : dict = df.describe(include='all').fillna("N/A") 
        description_model = {}
        for col_name, col_series in desc_raw.items():
            col_data : dict = col_series.to_dict()
            # Maping describe keys to Pydantic model field names (handle alias for %)
            model_data = {
                "count": col_data.get("count"), "mean": col_data.get("mean"),
                "std": col_data.get("std"), "min": col_data.get("min"),
                "q25": col_data.get("25%"), "q50": col_data.get("50%"),
                "q75": col_data.get("75%"), "max": col_data.get("max"),
                "unique": col_data.get("unique"), "top": col_data.get("top"),
                "freq": col_data.get("freq"), "dtype": str(df[col_name].dtype)
            }
            description_model[col_name] = DatasetColumnSummary(**model_data)

        summary = DatasetSummary(
            shape=df.shape,
            description=description_model,
            columns=df.columns.tolist(),
            total_transcripts=df['transcript_id'].nunique(),
            total_messages=len(df)
        )
        logger.info("Dataset summary generated successfully.")
        return summary
    except Exception as e:
        logger.error(f"Error generating dataset summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error generating summary: {str(e)}")


@router.post("/transform/message", response_model=ProcessedMessageOutput, tags=["Data Transformation"])
async def transform_raw_message(raw_message_input: RawMessageInputForTransform, app_state: dict = Depends(get_app_state)):    
    transformer = app_state["data_transformer"]
    try:
        processed_text = transformer.preprocess_text(raw_message_input.message)
        
        output = ProcessedMessageOutput(
            original_message=raw_message_input.message,
            processed_message_text=processed_text
        )
        logger.info(f"Message transformed successfully: {processed_text[:50]}...")
        return output

    except Exception as e:
        logger.error(f"Error transforming message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during message transformation: {str(e)}")


@router.post("/insights/transcript", response_model=InsightsOutput, tags=["LLM Insights"])
async def get_transcript_insights(transcript_input: TranscriptInput, app_state: dict = Depends(get_app_state)):
    llm_extractor : LLMInsightsExtractor = app_state["llm_insights_extractor"]
    sentence_model  = app_state["sentence_model"]
    faiss_index  = app_state["faiss_index"]
    faiss_url_map  = app_state["faiss_url_map"]
    try:
        transcript_messages_data = []
        full_transcript_text_for_search_query = "\n".join([f"{msg.agent}: {msg.message}" for msg in transcript_input.content])

        # 1. Generate the url search query
        search_query_llm = await llm_extractor.generate_search_query_from_transcript(full_transcript_text_for_search_query)
        logger.info(f"LLM generated search query: '{search_query_llm}'")

        # 2. Embed the search query
        query_embedding = sentence_model.encode([search_query_llm], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')

        # 3. Search FAISS index
        k_results = 1 
        distances, indices = faiss_index.search(query_embedding, k_results)
        top_match_index = indices[0][0]
        final_article_link = faiss_url_map.get(top_match_index)
        logger.info(f"FAISS top match (index {top_match_index}, dist {distances[0][0]}): {final_article_link}")

        for msg_input in transcript_input.content:
            transcript_messages_data.append({
                "article_url": final_article_link,
                "message": msg_input.message,
                "agent": msg_input.agent,
                "sentiment": msg_input.sentiment,
                "knowledge_source": tuple(msg_input.knowledge_source) if msg_input.knowledge_source else tuple(),
                "turn_rating": msg_input.turn_rating
            })
            
        transcript_df = pd.DataFrame(transcript_messages_data)
        insights : dict = await llm_extractor.get_transcript_summary_details(transcript_df)
        
        response_data = InsightsOutput(
            possible_article_link=insights.get("possible_article_link"),
            messages_agent_1=insights.get("messages_agent_1"),
            messages_agent_2=insights.get("messages_agent_2"),
            agent_1_baseline_sentiment_mode = insights.get("overall_sentiment_agent_1_baseline (mode)"),
            agent_2_baseline_sentiment_mode = insights.get("overall_sentiment_agent_2_baseline (mode)"),
            llm_overall_sentiment_agent_1=insights.get("llm_overall_sentiment_agent_1"),
            llm_agent_1_sentiment_summary=insights.get("llm_agent_1_sentiment_summary"),
            llm_overall_sentiment_agent_2=insights.get("llm_overall_sentiment_agent_2"),
            llm_agent_2_sentiment_summary=insights.get("llm_agent_2_sentiment_summary"),
            llm_summary_of_chat=insights.get("llm_summary_of_chat")
        )
        
        return response_data
    except Exception as e:
        logger.error(f"Error processing transcript for insights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing transcript: {str(e)}")