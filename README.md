## A "Soft API" to statistics about Denmark

### Introduction

AI is a solution to the increasing complexity of the modern world causing the rise of bureaucracy and information overload. - Rohit Krishnan

Todays large organisations are riddled with stringent rules and bureaucracy thet helps make processes more scalable. Ideally everyone would like a simpler system, where an agent made informed decisions on a case by case basis. That is how things used to be. Fewer rules and more agency.
AI gives us the possibility of creating a scalable version of what we used to have. A personal connection inside organisations that helps us make the right decisions.
In Rohit Krishnans words then you could think of this as a "soft API" that allows for more human ways of interacting with large organisations.

### Project

This code repo contains a quick first draft of a soft api to Denmarks Statistics. You can ask questions and the AI will try and fetch the correct dataset for you and plot it in a way that answers your questions. This is illustrated in the below video. Various ways of

<iframe width="560" height="315" src="https://www.youtube.com/embed/pBPuyM_DMk4?si=ZjIqSeOLXUoq5otE" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Since Denmarks statistics has a traditional hard API and a traditional graphical user interface with keyword search and various selection decisions then you can compare the different user interfaces.

 - Denmarks Statistics hard API
 - Denmark Statistics user interface to hard API
 - Denmarks Statistics soft api

### Code

The frontend is implemented using streamlit and for finding the right tables then table descriptions are embedded and stored in a postgres database with pgvector enabled for vector similarity search.
Cohere is used for embeddings and reranking and OpenAI GPT's are used for creating API calls, writing code and communicating with user.

The project is hosted on Replit and can be accessed here.

### TODOs
- [ ] Finetune custom embeddings and reranker for better table lookup.
- [ ] Finetune small llm to do simple tasks instead of using GPT-3.5-turbo.
- [ ] Implement intelligent caching. Store question/answer pairs and do similarity search on previous questions with some meaningful filtering with regards to table fetched etc.
- [ ] Finetune small llm to replace RAG framework. Should be possible to make small llm remember all 2300 tables publicly available and make better decisions than RAG.
