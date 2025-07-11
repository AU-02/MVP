import React, { useState, useEffect, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaUserCircle } from "react-icons/fa";
import "../themes/theme.css"; 

const Navbar = () => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const navigate = useNavigate();
  const dropdownRef = useRef(null);

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  return (
    <nav className="navbar">
      <div className="nav-left">
        <Link to="/" className="logo">D3-ND</Link>
      </div>
      <div className="nav-center">
        <Link to="/HomeScreen" className="nav-link">Home</Link>
        <Link to="/tab2" className="nav-link"></Link>
        <Link to="/tab3" className="nav-link"></Link>
      </div>
      <div className="nav-right">
        {/* Profile Icon with Fixed Dropdown */}
        <div className="profile-container" ref={dropdownRef}>
          <FaUserCircle 
            className="profile-icon" 
            onClick={() => setDropdownOpen(!dropdownOpen)}
          />
          {dropdownOpen && (
            <div className="dropdown">
              <button onClick={handleLogout}>Logout</button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;