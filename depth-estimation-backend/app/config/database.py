from motor.motor_asyncio import AsyncIOMotorClient

# DB Connection String
MONGO_URI = "mongodb+srv://anoukudumalagala:G4AmIjevwQmuKT3W@cluster0.peynz.mongodb.net/D3MSD?retryWrites=true&w=majority&appName=Cluster0&connectTimeoutMS=30000&serverSelectionTimeoutMS=10000"

# Connect to MongoDB Atlas
client = AsyncIOMotorClient(
    MONGO_URI,
    maxPoolSize=10,
    minPoolSize=2,
    maxIdleTimeMS=60000,
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=30000,
    heartbeatFrequencyMS=10000
)

database = client["D3MSD"]  
users_collection = database["users"] 

# DB Connection test function
async def test_connection():
    try:
        await client.admin.command('ping')
        print("MongoDB connection successful!")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False