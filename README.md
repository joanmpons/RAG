# RAG
## Extending ChatGPT capabilities through Retrieval Augmented Generation
*Python, NLP, RAG*
<p align="justify"> 
La finalidad del proyecto es facilitar la consulta de las ayudas, subvenciones, becas y premios ofrecidos por la administración mediante el uso de un chatbot. Un caso de uso sería, por ejemplo, alguien que tiene intención de realizar una reforma y quiere informarse de si puede beneficiarse de alguna ayuda pública para costear el proyecto. 
</p>

## Workflow 
La estructura simplificada de la solución es la siguiente:
  - Web scraping de los datos de la administración.
  - Embedding de la información obtenida.
  - Query a ChatGPT.
  - Análisis de cosine similarity para extraer las ayudas relevantes para la query.
  - Query enriquecida con el nuevo contexto a ChatGPT.
