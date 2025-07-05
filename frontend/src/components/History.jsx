// History.jsx
import React from "react";
import { Power } from "lucide-react";

const History = ({ isLoggedIn, isHistoryOpen, onClose }) => {
  return (
    <div className={`history-panel ${isHistoryOpen ? "open" : ""}`}>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="text-white m-0">Chat History</h5>
        <Power onClick={onClose} aria-label="Close" style={{ color: "fff", cursor:"pointer" }}/>
      </div>

      {isLoggedIn ? (
        <ul className="mt-4 list-unstyled text-white">
          <li>Conversation #1</li>
          <li>Conversation #2</li>
          <li>Conversation #3</li>
        </ul>
      ) : (
        <div className="text-white-50 mt-4">
          <p>
            Please <strong>sign in</strong> to access your chat history.
          </p>
        </div>
      )}
    </div>
  );
};

export default History;
