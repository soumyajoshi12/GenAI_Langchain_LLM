
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    
)

model = ChatHuggingFace(llm=llm)


print("****************if want to exit press 0******************")
while(True):
    prompt = input("You:")
    if prompt == '0':
            break
    response = model.invoke(prompt)
    print("\nAI:")
    print(response.content)
