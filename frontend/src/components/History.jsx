// History.jsx
import React from 'react';

const History = ({ isLoggedIn, isHistoryOpen }) => {
  return (
    <div className={`history-panel ${isHistoryOpen ? 'open' : ''}`}>  
      <h5 className="text-white">Chat History</h5>
      {isLoggedIn ? (
        <ul className="mt-5 list-unstyled text-white">
          <li>Conversation #1</li>
          <li>Conversation #2</li>
          <li>Conversation #3</li>
        </ul>
      ) : (
        <div className="text-white-50">
          <p>Please <strong>sign in</strong> to access your chat history.</p>
        </div>
      )}
    </div>
  ); 
};

export default History;
