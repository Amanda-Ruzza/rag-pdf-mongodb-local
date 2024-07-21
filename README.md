# Local Machine GenAI Chatbot application with Streamlit, MongoDB Atlas, Langchain and OpenAI

---

This Python Retrieval-Augmented Generation (RAG) application is able to read multiple PDFs - up to 200MB at a time - and answer questions based on the information in those PDFs. In simpler terms, it can find relevant information from the PDFs and use that information to answer your questions.

It was developed locally for future Cloud Deployment - in AWS and GCP - using Serverless Containers. Application stack:

* **Streamlit** - Front End
* **OpenAi** - LLM/Foundation Model
* **Langchain** - NLP Orchestration
* **MongoDB Atlas Vector Search** - Cloud-based Vector Database
* **Dotenv** - Local secret management
* **PyPDF** - PDF text extraction
* **PyTesseract** - OCR on AES Encrypted PDFs or PDFs with images in the background that would result in an empty text extraction

---

## Key Features

* Secure API/TOKEN keys connection hidden in the `.env` file
* Processes multiple files - up to 200MB - within 1 single upload operation
* Capability to answer questions based on documents that are already vectorized and stored in the database - no need to reupload the same PDFs
* Ability to extract text from AES locked PDFs or PDFs with background images that block a simple text extraction
* Text extraction parallel processing for  PDFs > 5MB for faster application performance
* A _'Clear Chat History'_ button
* A series of observability/logs features for future Cloud Development considerations:
  * A Langchain `callback` function that calculates 'OpenAi' token usage and prints it to a logger file. ![cost-screenshot](images/openai-token-usage-mdb-logs-screenshot.png) 
  * MongoDB operation specific logs recorded through the `pymongo` driver
  * A `script execution time` measurement functionality

</br>

![mdb-vector-screenshot-1](images/mdb-compass-screenshot-1.png)

## Prerequisites

* Python >=3.11
* Tesseract CLI
* OpenAI API Key
* MongoDB Atlas Cluster and Database - Instructions for the free MongoDB Atlas account, cluster and database set up can be found [here](https://www.mongodb.com/docs/atlas/getting-started/) .

## Reference Architecture
---
![architecture-diagram](images/local-rag-mdb-diagram.png)

## Setup instructions
---
### MongoDB Atlas Setup
---
Networking
connection string

#### Vector Search Index Creation
---
Navigate to Data Services > Your Cluster > Browse Collections > Atlas Search

{Select movies collection from sample_mflix database and copy and paste the JSON snippet below.}

[add the JSON]


#### Additional Setup

* Install the [tesseract cli](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html) in your local machine and add the `tesseract location path` to the `.env` file - `pytesseract` is a python package for `tesseract`, however, it works out of the tesseract cli locally installed
* 


#### Virtual Environment Activation

Create a `chatbot-app` virtual environment for your project:

And activate it:
`source chatbot-env/bin/activate`

#### Shell Script

As an option, edit the `sample_run_chatbot.sh` bash script with your local machine project directory, and run this script to either activate the venv and run streamlit, or in case the venv is activated, to run streamlit:

`./sample_run_chatbot.sh`

## Future Improvements

Implement PDF metadata extraction, and send a JSON file with the 'PDF Name' + 'file size' + 'date processed' into a different MongoDB database within the DB Cluster, and create a container in Streamlit so that the user can visualize a list of previously processed PDFs, for further 'conversation'.


* Create a 'Web URL Input' functionality, so that the user has the option to either upload a file or add a PDF web url.
* Create a 'document uploaded' metadata JSON file that will be sent into a NoSQL database so that there is a record of all the PDFs previously vectorized, so that the user can view a list of these PDFs and ask questions about them.
* Create a drop down box in the UI, so that the user can view these available PDF file names.
* Cloud Native Deployment on AWS and GCP.
