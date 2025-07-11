from fastapi import APIRouter, Depends
from app.core.security import get_current_user  # Middleware to verify JWT

router = APIRouter()

@router.get("/home")
async def home(user: dict = Depends(get_current_user)):
    """Protected Home Route - Requires JWT Token"""
    return {"message": f"Welcome, {user['email']}!", "user": user}
