import React, { useState, useEffect, useRef, useCallback } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPenToSquare, faStop, faSpinner } from "@fortawesome/free-solid-svg-icons";
import ChatInput from "./ChatInput";
import aivis from "../assets/ai-assistant.png";
import Typewriter from "./Typewriter";

const ChatInterface = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [useDocumentMode, setUseDocumentMode] = useState(false);
  const [lastUploadedFileName, setLastUploadedFileName] = useState(null);
  const [controller, setController] = useState(null);
  const [stopTyping, setStopTyping] = useState(false);

  const chatContainerRef = useRef(null);

  // Scroll to bottom on new message
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Abort any previous controller before creating new one
  const handleSend = useCallback(async () => {
    if (!input.trim()) return;

    // Abort previous if any
    if (controller) {
      controller.abort();
    }

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

    setStopTyping(false);
    const abortController = new AbortController();
    setController(abortController);

    setMessages((prev) => [...prev, userMsg, processingMsg]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          filename: lastUploadedFileName,
          document_mode: useDocumentMode,
        }),
        signal: abortController.signal,
      });

      const data = await response.json();

      const aiReply = {
        id: Date.now() + 1,
        type: "ai",
        text: data?.response || "Sorry, I couldn't generate a response.",
      };

      setMessages((prev) =>
        prev.map((msg) => (msg.id === "loading-spinner" ? aiReply : msg))
      );
    } catch (error) {
      console.error("Error:", error);
      const errorReply = {
        id: Date.now() + 2,
        type: "ai",
        text:
          error.name === "AbortError"
            ? "⚠️ Response stopped by user."
            : "⚠️ Something went wrong. Please try again.",
      };
      setMessages((prev) =>
        prev.map((msg) => (msg.id === "loading-spinner" ? errorReply : msg))
      );
    } finally {
      setLoading(false);
      setController(null);
    }
  }, [input, lastUploadedFileName, useDocumentMode, controller]);

  const handleFileSelect = useCallback(async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessages((prev) => [
          ...prev,
          { id: Date.now(), type: "user", text: `📎 Uploaded: ${file.name}` },
        ]);
        setLastUploadedFileName(file.name);
      } else {
        console.error("Upload failed:", data?.error || "Unknown error");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  }, []);

  const handleEdit = useCallback(
    (id) => {
      setMessages((prevMessages) => {
        const toEdit = prevMessages.find((msg) => msg.id === id);
        if (!toEdit) return prevMessages;
        setInput(toEdit.text);
        return prevMessages.filter((msg) => msg.id !== id);
      });
    },
    [setInput]
  );

  const handleStop = useCallback(() => {
    if (controller) {
      controller.abort();
      setController(null);
    }
    setStopTyping(true);
    setLoading(false);
  }, [controller]);

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
          marginBottom: 20, // fixed to number for consistency
        }}
      >
        <div className="chat-header text-center mb-4">
          <img src={aivis} alt="ai_visualiser" className="img-fluid visual_img" />
          <h2 className="grad_text">Ask Recallo</h2>
        </div>

        {messages.map((msg) => (
          <div key={msg.id} className={`chat-response ${msg.type}`}>
            {msg.type === "user" ? (
              <>
                <p style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}>
                  <strong style={{ display: "block", marginBottom: "10px" }}>You:</strong>
                  {msg.text}
                </p>
                <div className="message-actions d-flex mt-3">
                  <button
                    className="btn chat_ic me-2"
                    onClick={() => handleEdit(msg.id)}
                    aria-label="Edit message"
                  >
                    <FontAwesomeIcon icon={faPenToSquare} style={{ color: "#fff" }} />
                  </button>
                  <button className="btn chat_ic" onClick={handleStop} aria-label="Stop response">
                    <FontAwesomeIcon icon={faStop} style={{ color: "#fff" }} />
                  </button>
                </div>
              </>
            ) : (
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
                    <FontAwesomeIcon icon={faSpinner} spin /> Recallo is processing...
                  </div>
                ) : (
                  <Typewriter key={msg.id} text={msg.text} stop={stopTyping} />
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <ChatInput
        input={input}
        setInput={setInput}
        handleSend={handleSend}
        handleStop={handleStop}
        loading={loading}
        useDocumentMode={useDocumentMode}
        setUseDocumentMode={setUseDocumentMode}
        handleFileSelect={handleFileSelect}
      />
    </div>
  );
};

export default ChatInterface;
