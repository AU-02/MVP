import React, { useState, useEffect } from "react";
import { registerUser, loginUser } from "../api/auth"; // Import API calls
import { useNavigate } from "react-router-dom"
import { FaGoogle, FaApple, FaCircle } from "react-icons/fa";
import "../themes/theme.css";


const GetStarted = () => {
  const [showRegister, setShowRegister] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);

  const slides = [
    { text: "Diffusion-Driven Depth Estimation Using Multi-Spectral Data", img: "/assets/ImageSlide1.jpg" },
    { text: "Experience a New Way of Interaction", img: "/assets/ImageSlide2.jpg" },
    { text: "Bringing Simplicity to Your Daily Life", img: "/assets/ImageSlide3.jpeg" }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((prevIndex) => (prevIndex + 1) % slides.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container">
      {/* Left Section - Image and About */}
      <div className="left-section" style={{ backgroundImage: `url(${slides[activeIndex].img})` }}>
        <div className="left-overlay"></div>
        <div className="logo">D3-ND</div>
        <div className="left-content">
          <p className="slider-text">{slides[activeIndex].text}</p>

          {/* Bottom Dots */}
          <div className="slider-dots">
            {slides.map((_, index) => (
              <div key={index} className={`dot ${index === activeIndex ? "active" : ""}`}></div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Section - Login/Register */}
      <div className="auth-section">
        <div className="auth-card">
          {showRegister ? (
            <RegisterForm setShowRegister={setShowRegister} />
          ) : (
            <LoginForm setShowRegister={setShowRegister} />
          )}
        </div>
      </div>
    </div>
  );
};

// Login Form
const LoginForm = ({ setShowRegister }) => {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await loginUser(formData);
      localStorage.setItem("token", response.access_token); // Save JWT token
      setMessage("Login Successful!");
      navigate("/HomeScreen"); // Redirect to home
    } catch (error) {
      setMessage(error.detail || "Login failed");
    }
  };

  return (
    <div>
      <h2 className="form-title">Log in</h2>
      <form onSubmit={handleSubmit}>
        <input type="email" name="email" placeholder="Email" onChange={handleChange} required />
        <input type="password" name="password" placeholder="Enter your password" onChange={handleChange} required />
        <div className="terms-container">
          <input type="checkbox" className="checkbox" required />
          <label className="terms">
            I agree to the <span className="link">Terms & Conditions</span>
          </label>
        </div>
        <button className="submit-btn" type="submit">Log in</button>
      </form>
      {message && <p>{message}</p>}
      <p className="toggle-text">
        Don’t have an account?{" "}
        <span className="link" onClick={() => setShowRegister(true)}>Register Here</span>
      </p>
    </div>
  );
};

// Register Form
const RegisterForm = ({ setShowRegister }) => {
  const [formData, setFormData] = useState({
    full_name: "",
    email: "",
    password: "",
    confirm_password: "",
    terms_accepted: false,
  });
  const [message, setMessage] = useState("");

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({ ...formData, [name]: type === "checkbox" ? checked : value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await registerUser(formData);
      setMessage("Registration Successful!");
      setShowRegister(false); // Switch to login form
    } catch (error) {
      setMessage(error.detail || "Registration failed");
    }
  };

  return (
    <div>
      <h2 className="form-title">Create an account</h2>
      <p className="toggle-text">
        Already have an account?{" "}
        <span className="link" onClick={() => setShowRegister(false)}>Log in</span>
      </p>
      <form onSubmit={handleSubmit}>
        <div className="name-fields">
          <input type="text" name="full_name" placeholder="Full Name" onChange={handleChange} required />
        </div>
        <input type="email" name="email" placeholder="Email" onChange={handleChange} required />
        <input type="password" name="password" placeholder="Enter password" onChange={handleChange} required />
        <input type="password" name="confirm_password" placeholder="Confirm password" onChange={handleChange} required />
        <div className="terms-container">
          <input type="checkbox" name="terms_accepted" className="checkbox" onChange={handleChange} required />
          <label className="terms">
            I agree to the <span className="link">Terms & Conditions</span>
          </label>
        </div>
        <button className="submit-btn" type="submit">Create Account</button>
      </form>
      {message && <p>{message}</p>}
    </div>
  );
};

export default GetStarted;
