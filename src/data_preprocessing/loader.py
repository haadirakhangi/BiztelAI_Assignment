import pandas as pd
import json

class DataLoader:
    def __init__(self, file_path : str):
        self.file_path = file_path

    def load_json_dataset(self):
        try: 
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded JSON dataset from {self.file_path}")
            return data
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {self.file_path}")
            return None
        
    def structure_raw_dataset(self, raw_data : dict) -> pd.DataFrame:
        preprocessed_data = []
        for transcript_id, data in raw_data.items():
            article_url = data.get('article_url', '')
            config = data.get('config', '')
            conversation_rating_agent1 = data.get('conversation_rating', {}).get("agent_1")
            conversation_rating_agent2 = data.get('conversation_rating', {}).get("agent_2")

            for c in data.get("content", []):
                preprocessed_data.append({
                    "transcript_id": transcript_id,
                    "article_url": article_url,
                    "config": config,
                    "message": c.get("message"),
                    "agent": c.get("agent"),
                    "sentiment": c.get("sentiment"),
                    "knowledge_source": tuple(c.get("knowledge_source", [])),
                    "turn_rating": c.get("turn_rating"),
                    "conversation_rating_agent1": conversation_rating_agent1,
                    "conversation_rating_agent2": conversation_rating_agent2
                })

        df = pd.DataFrame(preprocessed_data)
        print("Raw dataset converted into DataFrame.")
        return df
