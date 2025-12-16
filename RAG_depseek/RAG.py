PDF_FILE = "duh.pdf"

# We'll be using Llama 3.1 8B for this example.
MODEL = "deepseek-r1:8b"

from langchain_community.document_loaders import PyPDFLoader

#1. Document reader
loader = PyPDFLoader(PDF_FILE)
pages = loader.load()

print(f"Number of pages: {len(pages)}")
print(f"Length of a page: {len(pages[1].page_content)}")
print("Content of a page:", pages[1].page_content)
# 1.2 Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

chunks = splitter.split_documents(pages)
print(f"Number of chunks: {len(chunks)}")
print(f"Length of a chunk: {len(chunks[1].page_content)}")
print("Content of a chunk:", chunks[1].page_content)
print("______________________splitter______________________________________________")
#
#
# # 2 Emberddings
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaEmbeddings
#
# embeddings = OllamaEmbeddings(model=MODEL)
# vectorstore = FAISS.from_documents(chunks, embeddings)
# # 2.1 REtriever
#
# retriever = vectorstore.as_retriever()
# retriever.invoke("What can you get away with when you only have a small number of user")
# print("______________________REtriever_______________________________________________")
#
# #_____________________________________________________________________________________________________________
# # 3. Model
# from langchain_ollama import ChatOllama
#
# model = ChatOllama(model=MODEL, temperature=0)
# msg =model.invoke("Who was the president of the United States in 2021?")
# print(msg.content)
#
# print("______________________model invoke______________________________________________")
#
#  ## 3.1 parser for better output
# from langchain_core.output_parsers import StrOutputParser
#
# parser = StrOutputParser()
#
# chain = model | parser
# print(chain.invoke("Who was the president of the United States in 2021?"))
# print("______________________parser______________________________________________")
#
# ###3.2 Prompt
# from langchain.prompts import PromptTemplate
#
# template = """
# You are an assistant that provides answers to questions based on
# a given context.
#
# Answer the question based on the context. If you can't answer the
# question, reply "I don't know".
#
# Be as concise as possible and go straight to the point.
#
# Context: {context}
#
# Question: {question}
# """
#
# prompt = PromptTemplate.from_template(template)
# print(prompt.format(context="Here is some context", question="Here is a question"))
#
# #3.2.1 Add prompt to the chain
#
# chain = prompt | model | parser
#
# print(chain.invoke({
#     "context": "Anna's sister is Susan",
#     "question": "Who is Susan's sister?"
# }))
# print("____________________________prompt to chain ")

#
# from operator import itemgetter
#
# chain = (
#     {
#         "context": itemgetter("question") | retriever,
#         "question": itemgetter("question"),
#     }
#     | prompt
#     | model
#     | parser
# )
#
#
# ### 44\. Ask Questions relates to dcuments
#
# questions = [
#     "What can you get away with when you only have a small number of users?",
#     "What's the most common unscalable thing founders have to do at the start?",
#     "What's one of the biggest things inexperienced founders and investors get wrong about startups?",
# ]
#
# for question in questions:
#     print(f"Question: {question}")
#     print(f"Answer: {chain.invoke({'question': question})}")
#     print("*************************\n")
#
#
#
