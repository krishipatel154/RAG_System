# RAG_System

# 1. What is Retrieval-Augmented Generation (RAG)?
RAG enhances the capabilities of language models by integrating them with a retrieval system. Instead of relying solely on the model’s internal knowledge, RAG allows the model to fetch relevant documents from a knowledge base in real-time. This integration enables the model to generate responses based on up-to-date and context-specific information, significantly improving the quality and relevance of the generated text.

<img width="686" height="292" alt="image" src="https://github.com/user-attachments/assets/5b8c86ca-288e-47f3-ae83-bce7a3af7a02" />

# 2. Embedding Model
In order to efficiently retrieve information, we need a way to represent documents in a format that can be processed mathematically. This is achieved through an embedding model, which translates content: text, images, videos, and audio into vectors that capture the semantic meaning of the data.

<img width="733" height="348" alt="image" src="https://github.com/user-attachments/assets/233c9311-3d16-4990-956b-c32b35f5d432" />

These vectors represent the position of each piece of data in a multi-dimensional space. The closeness of vectors indicates their similarity, This makes it easier to identify related content.


<img width="415" height="330" alt="image" src="https://github.com/user-attachments/assets/053dc7cb-7e23-4ae5-9cc1-77dc2ce02281" />

Embeddings allow for the calculation of similarities between data points, so when a user asks a question, the model can retrieve the most relevant content by comparing the vectorized query to the vectorized documents in the knowledge base.


# How to run the code?

## Step1: build the virtual environment
### command: python -m venv venv

## Step2: install the requirements
### command: pip install -r requirements.txt

## Step3: create .env file
### generate the groq api (https://groq.com/) and save it in .evn as, GROQ_API_KEY = YOUR_API_KEY

## Step4: run the code
### command: python app.py
