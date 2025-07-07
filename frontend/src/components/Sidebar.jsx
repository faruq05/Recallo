import React from "react";
import {
  FaTasks,
  FaBook,
  FaCheck,
  FaExclamation,
  FaChartBar,
  FaClipboardList,
  FaCog,
  FaUser,
  FaHistory,
} from "react-icons/fa";
import recalloLogo from "../assets/recallo.png";
import { Link, useLocation } from "react-router-dom"; // 👈 Import useLocation
import { EqualApproximately } from "lucide-react";

const menuItems = [
  { icon: <FaTasks />, label: "To-Do List", path: "/todo" },
  { icon: <FaBook />, label: "Your Resources", path: "/resources" },
  { icon: <FaCheck />, label: "Covered Topics", path: "/covered" },
  { icon: <FaExclamation />, label: "Weak Areas", path: "/weak-areas" },
  { icon: <FaChartBar />, label: "Progress", path: "/progress" },
  { icon: <FaClipboardList />, label: "Quizzes", path: "/quizzes" },
  { icon: <FaBook />, label: "Exam Tracker", path: "/exam-tracker" },
  { icon: <FaCog />, label: "Settings", path: "/settings" },
];

const Sidebar = ({ isOpen, toggleSidebar, toggleHistory, isHistoryOpen, isLoggedIn }) => {
  const location = useLocation(); // 👈 Current route

  const handleSidebarToggle = () => {
    toggleSidebar();
    if (isHistoryOpen) toggleHistory();
  };

  return (
    <div className={`sidebar ${isOpen ? "open" : "collapsed"}`}>
      <div className="sidebar-header d-flex align-items-center justify-content-between p-3">
        {isOpen && (
          <a href="/">
            <img
              src={recalloLogo}
              alt="Recallo Logo"
              className="img-fluid logo"
              style={{ maxWidth: "120px" }}
            />
          </a>
        )}
        <EqualApproximately
          onClick={handleSidebarToggle}
          className="toggle-icon sidebar-toggler"
          style={{ color: "#ffffff", cursor: "pointer", fontSize: "1.5rem" }}
        />
      </div>

      {isLoggedIn ? (
        <>
          <ul className="sidebar-menu">
            {menuItems.map((item, index) => {
              const isActive = location.pathname === item.path;
              return (
                <li
                  key={index}
                  className="menu-item"
                  style={{
                    backgroundColor: isActive ? "var(--cs-border)" : "transparent",
                    borderRadius: "8px",
                  }}
                >
                  <Link
                    to={item.path}
                    className="d-flex align-items-center text-white text-decoration-none w-100"
                  >
                    <span className="icon me-2">{item.icon}</span>
                    {isOpen && <span className="label">{item.label}</span>}
                  </Link>
                </li>
              );
            })}

            <li
              onClick={toggleHistory}
              className="d-flex align-items-center mb-3 text-white"
              style={{
                cursor: "pointer",
                backgroundColor: isHistoryOpen ? "var(--cs-border)" : "transparent",
                borderRadius: "8px",
              }}
            >
              <span className="icon me-2">
                <FaHistory />
              </span>
              {isOpen && <span className="label">History</span>}
            </li>
          </ul>

          <div className="profile_section">
            <li
              className="menu-item"
              style={{
                backgroundColor: location.pathname === "/profile" ? "var(--cs-border)" : "transparent",
                borderRadius: "8px",
              }}
            >
              <Link
                to="/profile"
                className="d-flex align-items-center text-white text-decoration-none w-100"
              >
                <span className="icon me-2">
                  <FaUser />
                </span>
                {isOpen && <span className="label">Profile</span>}
              </Link>
            </li>
          </div>
        </>
      ) : (
        isOpen && (
          <div className="sidebar-login-message text-center mt-4">
            <p className="mb-4 text-start">
              Recallo is much more than that. <br />
              <strong>Sign in to start your study journey!</strong>
            </p>
            <div className="d-flex flex-column gap-2">
              <Link to="/signin" className="btn btn-cs">
                Log In
              </Link>
              <Link to="/signin" className="btn btn-cs">
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
