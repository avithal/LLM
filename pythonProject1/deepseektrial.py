from langchain_ollama import ChatOllama
llm =ChatOllama(model="deepseek-r1:8b", temperature=0)
llm_json_mode =ChatOllama(model="deepseek-r1:8b", temperature=0, format="json") ## no think tokens
## with think token
# msg=llm.invoke("what is the capital of Israel")
# print(msg.content)

# without think token
import json
p =""" your goal is to generate targeted web search query.
The query will gather information related to a specific topic.
Topic : cats
REturn your query as a Json object
{{
 ""query":"string",
 "aspect":"string",
 "rationale":"string
 }}
 """
msg = llm_json_mode.invoke(p)
query = json.loads(msg.content)
print(query)