from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple

class Message(BaseModel):
    message: str
    agent: str
    sentiment: Optional[str] = None
    knowledge_source: Optional[List[str]] = Field(default_factory=list)
    turn_rating: Optional[str] = None

class TranscriptInput(BaseModel):
    content: List[Message] = Field(..., min_length=1, description="List of messages in the transcript") 

# --- Endpoint 1: Processed Dataset Summary ---
class DatasetColumnSummary(BaseModel):
    count: Optional[Any] = None
    mean: Optional[Any] = None
    std: Optional[Any] = None
    min: Optional[Any] = None
    q25: Optional[Any] = Field(None, alias="25%") 
    q50: Optional[Any] = Field(None, alias="50%")
    q75: Optional[Any] = Field(None, alias="75%")
    max: Optional[Any] = None
    dtype: Optional[str] = None
    unique: Optional[Any] = None
    top: Optional[Any] = None
    freq: Optional[Any] = None

class DatasetSummary(BaseModel):
    shape: Tuple[int, int]
    description: Dict[str, DatasetColumnSummary] # For df.describe() output, more structured
    columns: List[str]
    total_transcripts: int
    total_messages: int

# --- Endpoint 2: Real-time Data Transformation ---
class RawMessageInputForTransform(BaseModel): 
    message: str

class ProcessedMessageOutput(BaseModel):
    original_message: str
    processed_message_text: Optional[str] = None


# --- Endpoint 3: Processed Insights ---
class InsightsOutput(BaseModel):
    possible_article_link: str
    messages_agent_1: int
    messages_agent_2: int
    agent_1_baseline_sentiment_mode: str
    agent_2_baseline_sentiment_mode: str
    llm_overall_sentiment_agent_1: str
    llm_agent_1_sentiment_summary: str
    llm_overall_sentiment_agent_2: str
    llm_agent_2_sentiment_summary: str
    llm_summary_of_chat: str