import React, { useState, useEffect, useRef } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faPaperPlane,
  faPenToSquare,
  faStop,
  faSpinner,
} from "@fortawesome/free-solid-svg-icons";
import FileUpload from "./FileUpload";
import aivis from "../assets/ai-assistant.png";
import ReactMarkdown from "react-markdown";
import TextareaAutosize from "react-textarea-autosize";
import Typewriter from "./Typewriter";

const ChatInterface = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (input.trim() === "") return;

    const userMsg = {
      id: Date.now(),
      type: "user",
      text: input,
    };

    const processingMsg = {
      id: "loading-spinner",
      type: "ai",
      text: "Recallo is processing...",
      isProcessing: true,
    };

    setMessages((prev) => [...prev, userMsg, processingMsg]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();

      const aiReply = {
        id: Date.now() + 1,
        type: "ai",
        text: data.response || "Sorry, I couldn't generate a response.",
      };

      // Replace the spinner with real response
      setMessages((prev) =>
        prev.map((msg) => (msg.id === "loading-spinner" ? aiReply : msg))
      );
    } catch (error) {
      console.error("Error:", error);
      const errorReply = {
        id: Date.now() + 2,
        type: "ai",
        text: "⚠️ Something went wrong. Please try again.",
      };
      setMessages((prev) =>
        prev.map((msg) => (msg.id === "loading-spinner" ? errorReply : msg))
      );
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (file) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        type: "user",
        text: `📎 Uploaded: ${file.name}`,
      },
    ]);
  };

  const handleEdit = (id) => {
    const toEdit = messages.find((msg) => msg.id === id);
    if (toEdit) {
      setInput(toEdit.text);
      setMessages(messages.filter((msg) => msg.id !== id));
    }
  };

  const handleStop = () => {
    alert("Stop action triggered.");
  };

  return (
    <div
      className="chatinterface"
      style={{ height: "100vh", display: "flex", flexDirection: "column" }}
    >
      <div
        className="chat-response-section"
        ref={chatContainerRef}
        style={{
          flexGrow: 1,
          overflowY: "auto",
          padding: "10px",
          marginBottom: "50px",
        }}
      >
        <div className="chat-header text-center mb-4">
          <img
            src={aivis}
            alt="ai_visualiser"
            className="img-fluid visual_img"
          />
          <h2 className="grad_text">Ask Recallo</h2>
        </div>

        {messages.map((msg) => (
          <div key={msg.id} className={`chat-response ${msg.type}`}>
            {msg.type === "user" ? (
              <>
                {/* user response */}
                <p style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}>
                  <strong style={{ display: "block", marginBottom: "10px" }}>
                    You:
                  </strong>
                  {msg.text}
                </p>
                <div className="message-actions d-flex mt-3">
                  <button
                    className="btn chat_ic me-2"
                    onClick={() => handleEdit(msg.id)}
                  >
                    <FontAwesomeIcon
                      icon={faPenToSquare}
                      style={{ color: "#ffffff" }}
                    />
                  </button>
                  <button className="btn chat_ic" onClick={handleStop}>
                    <FontAwesomeIcon
                      icon={faStop}
                      style={{ color: "#ffffff" }}
                    />
                  </button>
                </div>
              </>
            ) : (
              // ai response
              <div>
                <strong className="d-flex align-items-center mb-3 ai_response">
                  <img
                    src={aivis}
                    alt="ai_visualiser"
                    className="img-fluid visual_img rec_img"
                  />
                  Recallo:
                </strong>
                {msg.isProcessing ? (
                  <div className="processing-spinner">
                    <FontAwesomeIcon icon={faSpinner} spin /> Recallo is
                    processing...
                  </div>
                ) : (
                  <Typewriter key={msg.id} text={msg.text} />
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="chat-input-section" style={{ padding: "10px" }}>
        <FileUpload onFileSelect={handleFileSelect} />
        <TextareaAutosize
          rows={1}
          placeholder="Type your question here..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault(); // Prevent newline
              handleSend(); // Send message
            }
          }}
          style={{ width: "100%", padding: "10px", whiteSpace: "pre-wrap" }} // Preserve formatting
        />
        <button className="send-button btn btn-cs" onClick={handleSend}>
          {loading ? (
            <FontAwesomeIcon icon={faSpinner} spin />
          ) : (
            <FontAwesomeIcon icon={faPaperPlane} />
          )}
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;
