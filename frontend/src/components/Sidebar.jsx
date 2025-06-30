import React from "react";
import {
  FaBars,
  FaTasks,
  FaBook,
  FaCheck,
  FaExclamation,
  FaChartBar,
  FaClipboardList,
  FaCog,
  FaUser,
} from "react-icons/fa";
import recalloLogo from "../assets/recallo.png";
import { Link } from "react-router-dom";

const menuItems = [
  { icon: <FaTasks />, label: "To-Do List" },
  { icon: <FaBook />, label: "Your Resources" },
  { icon: <FaCheck />, label: "Covered Topics" },
  { icon: <FaExclamation />, label: "Weak Areas" },
  { icon: <FaChartBar />, label: "Progress" },
  { icon: <FaClipboardList />, label: "Quizzes" },
  { icon: <FaBook />, label: "Exam Tracker" },
  { icon: <FaCog />, label: "Settings" },
];

const Sidebar = ({ isOpen, toggleSidebar, isLoggedIn }) => {
  return (
    <div className={`sidebar ${isOpen ? "open" : "collapsed"}`}>
      <div className="sidebar-header">
        {isOpen && (
          <a href="/"><img
            src={recalloLogo}
            alt="Recallo Logo"
            className="img-fluid logo"
            style={{ maxWidth: "150px" }}
          /></a>
        )}
        <FaBars
          onClick={toggleSidebar}
          style={{ color: "#ffffff", cursor: "pointer", fontSize: "1.5rem" }}
        />
      </div>

      {isLoggedIn ? (
        <>
          <ul className="sidebar-menu">
            {menuItems.map((item, index) => (
              <li key={index}>
                <span className="icon">{item.icon}</span>
                {isOpen && <span className="label">{item.label}</span>}
              </li>
            ))}
          </ul>

          <div className="profile_section">
            <li>
              <span className="icon">
                <FaUser />
              </span>
              {isOpen && <span className="label">Profile</span>}
            </li>
          </div>
        </>
      ) : (
        isOpen && (
          <div
            className={`sidebar-login-message text-center mt-4 ${
              isOpen ? "visible" : "hidden"
            }`}
          >
            <p className="mb-4 text-start">
              Recallo is much more than that. <br />
              <strong>Sign in to start your study journey!</strong>
            </p>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "10px",
          
              }}
            >
              <Link to="/signin" className="btn btn-cs">
                Log In
              </Link>
              <Link to="/signin" className="btn btn-cs ">
                Sign Up
              </Link>
            </div>
          </div>
        )
      )}
    </div>
  );
};

export default Sidebar;
