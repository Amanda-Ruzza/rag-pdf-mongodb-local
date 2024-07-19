# Local Machine GenAI Chatbot application with Streamlit, MongoDB Atlas, Langchain, and OpenAI
---
A PDF Chatbot developed locally for future Cloud deployment using Serverless Containers

{finish this } showcasing a reference architecture for building a production-ready Generative AI application. this architecture leverages:

Streamlit: A framework for rapid development of interactive, real-time data-driven web applications.
MongoDB Atlas: A scalable, cloud-based database for efficient storage and retrieval of unstructured data, crucial for training and running AI models.
OpenAI: A foundational model [FINISH]
Langchain: A tool for seamless integration of complex natural language processing pipelines, enhancing the application's ability to understand and generate human-like text.


## Key Features
---
Ability to process multiple PDFs at the same time, up to 200MB
A `clear chat history` button for user privacy
OCR capabilities for AES encrypted and or watermarked PDFs
text extraction parallel processing for  PDFs > 5MB
OpenAI token usage and cost printed to a `.log` file for pre-production cost estimation
Application specific logs
MongoDB operation specific logs


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
Implement PDF metadata extraction, and send a JSON file with the 'PDF Name' + 'file size' + 'date processed' into a different MongoDB database within the DB Cluster, and create a container in Streamlit so that the user can visualize a list of previously processed PDFs, for further 'conversation'.
