import axios from "axios";

const API_URL = "http://127.0.0.1:8000/auth"; // FastAPI backend URL

// Register User
export const registerUser = async (userData) => {
  try {
    const response = await axios.post(`${API_URL}/register`, userData);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Login User
export const loginUser = async (loginData) => {
  try {
    const response = await axios.post(`${API_URL}/login`, loginData);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};
