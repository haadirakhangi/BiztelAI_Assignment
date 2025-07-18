from groq import AsyncGroq 
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import asyncio 
import json 

load_dotenv(find_dotenv())

class LLMInsightsExtractor:
    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

        self.sentiment_system_prompt = "You are a data analysis API that performs sentiment analysis on text. Respond only with JSON using this format: {\"agent_1\": {\"summary\": \"One sentence summary of the overall sentiment\", \"sentiment\": \"Curious to dive deeper | Neutral | Happy | Surprised | Disgusted | Sad | Angry | Fearful\"}, \"agent_2\": {\"summary\": \"One sentence summary of the overall sentiment\", \"sentiment\": \"Curious to dive deeper | Neutral | Happy | Surprised | Disgusted | Sad | Angry | Fearful\"}}"

        self.sentiment_base_user_prompt = """Analyze the following set of messages and determine the overall sentiment expressed by the agent. Choose only one from the following sentiments: Curious to dive deeper, Neutral, Happy, Surprised, Disgusted, Sad, Angry, Fearful. Messages: """

        self.summarization_prompt_base = "Summarize the following conversation concisely:\n"

        self.search_query_system_prompt = "You are an expert at identifying the main topic of a conversation and generating a concise search query to find a news article about that topic. The article is likely from the Washington Post. Respond only with JSON using this format: {\"search_query\": \"Search query here\"}"
        self.search_query_user_prompt_template = """Given the following conversation transcript, what is the primary news article topic being discussed? Generate a short, effective search query (3-7 words) that could be used to find this article on a news website like the Washington Post. Focus on key entities, events, or subjects.\nConversation:\n{transcript_text}\nSearch Query:"""
        
        # Default response structure in case of LLM errors
        self.default_sentiment_response = {
            "sentiment_analysis": {
                "summary": "Could not determine sentiment due to an error or no messages.",
                "sentiment": "N/A"
            }
        }
        self.default_summary_response = "Could not generate summary due to an error or no transcript."
        self.default_search_query = "general news topic"        

    async def _get_llm_sentiment(self, messages_text: str) -> dict:
        """Helper function to get sentiment for a given text block using LLM."""
        full_prompt = self.sentiment_base_user_prompt + messages_text
        try:
            response = await self.client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[
                    {"role": "system", "content": self.sentiment_system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError in sentiment analysis: {e}")
            return self.default_sentiment_response
        except Exception as e:
            print(f"Error in LLM sentiment call: {e}")
            return self.default_sentiment_response
        
    async def generate_search_query_from_transcript(self, transcript_text: str) -> str:
        """
        Uses LLM to generate a search query from the transcript text.
        """
        user_prompt = self.search_query_user_prompt_template.format(transcript_text=transcript_text)
        try:
            response = await self.client.chat.completions.create(
                model="gemma2-9b-it", 
                messages=[
                    {"role": "system", "content": self.search_query_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=20,
                response_format={"type": "json_object"}
            )
            query = response.choices[0].message.content.strip()
            return json.loads(query)
        except Exception as e:
            print(f"Error in LLM search query generation: {e}")
            return self.default_search_query

    async def _get_llm_summary(self, full_transcript_text: str) -> str:
        """Helper function to get a summary for the entire transcript using LLM."""            
        full_prompt = self.summarization_prompt_base + full_transcript_text
        try:
            response = await self.client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM summary call: {e}")
            return self.default_summary_response

    async def get_transcript_summary_details(self, transcript_data: pd.DataFrame) -> dict:
        """
        Asynchronously processes a transcript DataFrame to extract details,
        including LLM-based sentiment analysis and chat summarization.
        """

        # 1. Possible article link
        article_link = transcript_data['article_url'].iloc[0]

        # 2. Number of messages sent by agent 1 and agent 2
        message_counts = transcript_data['agent'].value_counts()
        messages_agent_1 = int(message_counts.get('agent_1', 0))
        messages_agent_2 = int(message_counts.get('agent_2', 0))

        # 3 and 4. For overall sentiment per agent 
        # Baseline Sentiment using mode:
        sentiment_agent_1_mode = transcript_data[transcript_data['agent'] == 'agent_1']['sentiment'].mode()
        overall_sentiment_agent_1_baseline = sentiment_agent_1_mode[0] if not sentiment_agent_1_mode.empty else "N/A"

        sentiment_agent_2_mode = transcript_data[transcript_data['agent'] == 'agent_2']['sentiment'].mode()
        overall_sentiment_agent_2_baseline = sentiment_agent_2_mode[0] if not sentiment_agent_2_mode.empty else "N/A"

        # Sentiment analysis per agent using LLM:              
        full_transcript_text = "\n".join([f"{row['agent']}: {row['message']}" for _, row in transcript_data.iterrows()])

        tasks = []
        tasks.append(self._get_llm_sentiment(full_transcript_text))
        tasks.append(self._get_llm_summary(full_transcript_text))
        results = await asyncio.gather(*tasks)
        
        llm_sentiment_response = results[0]
        llm_chat_summary_content = results[1]

        return {
            "possible_article_link": article_link,
            "messages_agent_1": messages_agent_1,
            "messages_agent_2": messages_agent_2,
            "overall_sentiment_agent_1_baseline (mode)": overall_sentiment_agent_1_baseline,
            "overall_sentiment_agent_2_baseline (mode)": overall_sentiment_agent_2_baseline,
            "llm_overall_sentiment_agent_1": llm_sentiment_response["agent_1"]["sentiment"],
            "llm_agent_1_sentiment_summary": llm_sentiment_response["agent_1"]["summary"],
            "llm_overall_sentiment_agent_2": llm_sentiment_response["agent_2"]["sentiment"],
            "llm_agent_2_sentiment_summary": llm_sentiment_response["agent_2"]["summary"],
            "llm_summary_of_chat": llm_chat_summary_content
        }
