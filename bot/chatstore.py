import yaml
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
import os

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class ChatHistoryHandler:
    def __init__(self):
        self.client = firestore.Client(project=config["google-firestore"]["PROJECT_ID"])
    
    def fetch_chat_history(self, chat_session_id: str):
        return FirestoreChatMessageHistory(
            session_id=chat_session_id,
            collection=config["google-firestore"]["COLLECTION_NAME"],
            client=self.client
        )
    
    def add_query(self, chat_session_id: str, query_text: str):
        chat_history = self.fetch_chat_history(chat_session_id)
        chat_history.add_user_message(query_text)
    
    def add_response(self, chat_session_id: str, response_text: str):
        chat_history = self.fetch_chat_history(chat_session_id)
        chat_history.add_ai_message(response_text)
    
    def get_chat_history(self, chat_session_id: str):
        chat_history = self.fetch_chat_history(chat_session_id)
        return chat_history.messages
    
    def delete_chat_history(self, chat_session_id: str):
        chat_history = self.fetch_chat_history(chat_session_id)
        chat_history.clear()