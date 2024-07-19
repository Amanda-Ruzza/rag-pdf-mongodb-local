"""This script pings the MongoDB Database with the environemnt variables to test the connection"""


from pymongo import MongoClient
from os import getenv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
atlas_uri = getenv("ATLAS_URI")
mongodb_db = getenv("MONGODB_DB")
mongodb_collection = getenv("MONGODB_COLLECTION")

# Debug print statements
print(f"ATLAS_URI: {atlas_uri}")
print(f"MONGODB_DB: {mongodb_db}")
print(f"MONGODB_COLLECTION: {mongodb_collection}")

# Create MongoDB client
client = MongoClient(atlas_uri)
db = client[mongodb_db]
collection = db[mongodb_collection]

# Test MongoDB connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

