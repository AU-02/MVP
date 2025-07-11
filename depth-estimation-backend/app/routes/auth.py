from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from app.config.database import users_collection, client
from app.core.security import hash_password, verify_password, create_access_token
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic Models
class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    confirm_password: str
    terms_accepted: bool

class UserLogin(BaseModel):
    email: EmailStr
    password: str

async def ensure_connection():
    """Ensure database connection is alive"""
    try:
        await client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Database connection unavailable. Please try again in a moment."
        )

@router.post("/register")
async def register(user: UserCreate):
    """Handles user registration with connection checking."""
    try:
        # Test connection first
        await ensure_connection()
        
        # Check if user already exists
        existing_user = await users_collection.find_one({"email": user.email})

        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        if user.password != user.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")

        if not user.terms_accepted:
            raise HTTPException(status_code=400, detail="You must accept the terms and conditions")

        # Hash password and create user
        hashed_password = hash_password(user.password)
        
        user_data = {
            "full_name": user.full_name,
            "email": user.email,
            "password": hashed_password,
            "terms_accepted": user.terms_accepted
        }

        # Insert user into database
        new_user = await users_collection.insert_one(user_data)
        
        logger.info(f"New user registered: {user.email}")
        return {
            "message": "User registered successfully", 
            "user_id": str(new_user.inserted_id)
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Registration failed for {user.email}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Registration failed. Please try again."
        )

@router.post("/login")
async def login(user: UserLogin):
    """Handles user login with connection checking."""
    try:
        # Test connection first
        await ensure_connection()
        
        # Find user in database
        existing_user = await users_collection.find_one({"email": user.email})

        if not existing_user:
            logger.warning(f"Login attempt with non-existent email: {user.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Verify password
        if not verify_password(user.password, existing_user["password"]):
            logger.warning(f"Invalid password attempt for: {user.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Create access token
        access_token = create_access_token(data={"sub": str(existing_user["_id"])})

        logger.info(f"Successful login for: {user.email}")
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": str(existing_user["_id"]),
                "email": existing_user["email"],
                "full_name": existing_user["full_name"]
            }
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Login failed for {user.email}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Login failed. Please try again."
        )

@router.get("/test-db")
async def test_database():
    """Test endpoint to check database connection"""
    try:
        await ensure_connection()
        
        # Try to count users
        user_count = await users_collection.count_documents({})
        
        return {
            "status": "Database connection successful",
            "user_count": user_count,
            "collection": "users"
        }
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}"
        )