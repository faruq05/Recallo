import React, { useState, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import ChatInterface from "../components/ChatInterface";
import History from "../components/History";
import { EqualApproximately } from "lucide-react";

const Chat = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth > 767);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);
  const toggleHistory = () => setIsHistoryOpen((prev) => !prev);

  const isLoggedIn = localStorage.getItem("userToken") !== null;

  useEffect(() => {
    const handleResize = () => {
      setIsSidebarOpen(window.innerWidth > 767);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="chat chat-wrapper d-flex min-vh-100">
      <div className={`sidebar-area ${isSidebarOpen ? "open" : "collapsed"}`}>
        <Sidebar
          isOpen={isSidebarOpen}
          toggleSidebar={toggleSidebar}
          toggleHistory={toggleHistory}
          isHistoryOpen={isHistoryOpen} 
          isLoggedIn={isLoggedIn}
        />
        <History isLoggedIn={isLoggedIn} isHistoryOpen={isHistoryOpen} />
      </div>

      <div className="chat-content flex-grow-1 position-relative">
        <ChatInterface />
        <span className="navbar-toggler-menu">
          <EqualApproximately
            className="d-lg-none position-fixed top-0 start-0 m-3"
            onClick={toggleSidebar}
            style={{ zIndex: 99 }}
          />
        </span>
      </div>
    </div>
  );
};

export default Chat;
