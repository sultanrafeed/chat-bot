import os
import yaml
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI  # Updated import as recommended
#from langchain.chains import ConversationChain  # Consider checking migration guide for alternative usage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

openai_api_key = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")

class ChatbotAgent:
    def __init__(self):
        # Use "history" to match the prompt expectation
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.chat_model = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
    #   self.chain = ConversationChain(llm=self.chat_model, memory=self.memory, verbose=True)
        self.chain = RunnableWithMessageHistory(
    runnable=self.chat_model,
    get_session_history=lambda sid: self.memory.load_memory(sid)
)
    def get_response(self, session_id: str, query: str) -> str:
        # Optionally, you can manage/reset memory based on the session_id.
        return self.chain.predict(input=query)