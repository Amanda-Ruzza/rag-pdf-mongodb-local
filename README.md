# rag-aws-ecs
---
A PDF Chatbot for AWS Deployment in Serverless Containers - Fargate-ECS

{finish this } showcasing a reference architecture for building a production-ready Generative AI application. this architecture leverages:

MongoDB Atlas: A scalable, cloud-based database for efficient storage and retrieval of unstructured data, crucial for training and running AI models.
Amazon Bedrock: A managed service for building, training, and deploying machine learning models at scale, streamlining generative AI workflows.
Langchain: A tool for seamless integration of complex natural language processing pipelines, enhancing the application's ability to understand and generate human-like text.
Streamlit: A framework for rapid development of interactive, real-time data-driven web applications.

## Key Features
---
OCR capabilities for AES encrypted and or watermarked PDFs


## Prerequisites
---
MongoDB Atlas Cluster and Database - Deploy a free MongoDB Atlas Cluster {add link}

AWS Account AWS Free Tier

## Reference Architecture
---


## Setup instructions
---

### MongoDB Atlas Setup
---


#### Vector Search Index Creation
---
Navigate to Data Services > Your Cluster > Browse Collections > Atlas Search

{Select movies collection from sample_mflix database and copy and paste the JSON snippet below.}

[add the JSON]

#### venv activation

`source chatbot-env/bin/activate`

#### shell script
run this script to either activate the venv and run streamlit, or in case the venv is activated, to run streamlit

`./run_chatbot.sh`

## Future Improvements
Implement OCR and PDF PDF and OCR text extraction parallel processing for fastest performance while uploading large PDFs
