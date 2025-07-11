from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId
from jose import JWTError, jwt
from app.config.database import users_collection
from typing import Dict
import os

# Load environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hashes a password."""
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    """Verifies a password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Generates a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# OAuth2 scheme to extract token from request
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Verifies JWT token and returns user info."""
    
    if not token:
        raise HTTPException(status_code=401, detail="Token is missing")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return {"email": user["email"], "id": str(user["_id"])}

    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid token")
