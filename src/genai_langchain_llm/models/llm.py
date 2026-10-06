
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    
)

model = ChatHuggingFace(llm=llm)

response = model.invoke(
        "Explain what an LLM is in simple words."
)

print("\nAI:")
print(response.content)

