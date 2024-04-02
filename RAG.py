# -*- coding: utf-8 -*-
#%% Libraries
import os
import time
import xmltodict
import numpy as np
# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
#Chat GPT API
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
# Web scraping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

#%% Config Web Scraping
class Browser:
    def __init__(self):
        self.service = Service()
        self.browser = webdriver.Chrome(service=self.service)
        self.timeout = 10
        self.wait = WebDriverWait(self.browser, self.timeout)

    def open_page(self, url:str):
        self.browser.get(url)

    def close_browser(self):
        self.browser.close()

    def add_input(self, by:By, value:str, text:str):
        field = self.browser.find_element(by=by, value=value)
        field.send_keys(text)
        time.sleep(1)

    def Keys_input(self, by:By, value:str, text:str):
           field = self.browser.find_element(by=by, value=value)
           field.send_keys(text)
           time.sleep(1)

    def click_button(self, by:By, value:str):
        button = self.browser.find_element(by=by, value=value)
        button.click()
        time.sleep(1)

    def scrape_res (self):
        self.click_button(by=By.XPATH, value='//*[@id="busquedaEmpleoBecasBean"]/section/div/div/div/button')
        self.wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="busquedaEmpleoBecasBean"]/section[2]/div/div/div[3]/button')))
        self.click_button(by=By.XPATH, value='//*[@id="busquedaEmpleoBecasBean"]/section[2]/div/div/div[3]/button')
        os.rename(r"empleoResultados_Busqueda_Becas.xml", r"empleoResultados_Busqueda_Becas.xml")

#%% Run scraping
if __name__ == '__main__':
    data_dict = {}
    browser = Browser()
    time.sleep(1)
    browser.open_page("https://administracion.gob.es/pagFront/empleoBecas/becasAyudasPremios/buscadorBecas.htm")
    browser.scrape_res()
    time.sleep(2)
    browser.close_browser()

#%% Parse xml
with open("empleoResultados_Busqueda_Becas.xml") as xml_file:
        data_dict = xmltodict.parse(xml_file.read())

data_dict = data_dict['BECAS']['BECA']

mylist = list()
nl = '\n'

for i in data_dict:
  text = str()
  for j,k in i.items():
    text += f'{j}:{k}.{nl}'
  mylist.append(text)

#%% Model config
# Path to model
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
# Model config
model_kwargs = {'device':'cpu'}
# Encoding config
encode_kwargs = {'normalize_embeddings': False}

#%% Initialize model
embeddings = HuggingFaceEmbeddings(
    model_name = modelPath,     
    model_kwargs = model_kwargs, 
    encode_kwargs = encode_kwargs 
)
# Create list with embedded context
context = []
for i in mylist:
  context.append(embeddings.embed_query(i))

#%% Chat GPT query with RAG
# Get API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Define chatOpenAI object with model specifications
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

# Initial message
message = [
    SystemMessage(content = ''),
    AIMessage(content = ''),
    HumanMessage(content = '')
    ]

# Appending initial message and response for converstional context
response = chat(message)
print(response.content)
message.append(response)

# User query
query = HumanMessage(content = 'Necesito rehabilitar un edificio antiguo')
message.append(query)
# Generate query embedding
query_result = embeddings.embed_query(query)

# Cosine similarity search
similarity = []
for i in context:
  similarity.append(np.dot(query_result,i)/np.linalg.norm(query_result)*np.linalg.norm(i))

# Provide new informational context by using the scraped information and define similarity threshold
results_index = [similarity.index(x) for x in similarity if x > np.quantile(similarity,0.99)]
for i in results_index:
    chat_context.append(mylist[i])
chat_context = "\n".join(chat_context)

# Generate the augmented query with the new information
augmented_query = f"""Using the contexts below, answer the query.

Contexts:
{chat_context}

Query: {query.content}"""

# Send Augmented query with conversational and informational context
query = HumanMessage(
    content=augmented_query
)
message.append(query)
response = chat(message)

# Get ChatGPT RAG response
print(response.content)


