from pymongo import MongoClient
from bson import ObjectId
import os
# MongoDB client setup

client = MongoClient(os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017"))
db = client["neurosattva"]
