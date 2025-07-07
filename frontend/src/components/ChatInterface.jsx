import React, { useState, useEffect, useRef } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faPenToSquare,
  faStop,
  faSpinner,
} from "@fortawesome/free-solid-svg-icons";
import aivis from "../assets/ai-assistant.png";
import Typewriter from "./Typewriter";
import ChatInput from "./ChatInput";
import { createClient } from "@supabase/supabase-js";

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_KEY
);

const ChatInterface = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [controller, setController] = useState(null);
  const [userId, setUserId] = useState(null);
  const chatContainerRef = useRef(null);
  const [useDocumentMode, setUseDocumentMode] = useState(false);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    // Get Supabase session and extract user_id
    const fetchSession = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (session?.user?.id) {
        setUserId(session.user.id);
      }
    };

    fetchSession();
  }, []);

  const handleSend = async () => {
    if (input.trim() === "") return;

    if (controller) controller.abort();
    const abortController = new AbortController();
    setController(abortController);

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
      const response = await fetch(
        `http://127.0.0.1:5000/${useDocumentMode ? "ask" : "chat"}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: input, user_id: userId }),
          signal: abortController.signal,
        }
      );

      const data = await response.json();
      console.log("\ud83d\udce9 Full API Response:", data);

      const aiReply = {
        id: Date.now() + 1,
        type: "ai",
        text: data.response || "Sorry, I couldn't generate a response.",
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
            ? "\u26a0\ufe0f Response stopped by user."
            : "\u26a0\ufe0f Something went wrong. Please try again.",
      };
      setMessages((prev) =>
        prev.map((msg) => (msg.id === "loading-spinner" ? errorReply : msg))
      );
    } finally {
      setLoading(false);
      setController(null);
    }
  };

  const handleFileSelect = (file) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        type: "user",
        text: `\ud83d\udcce Uploaded: ${file.name}`,
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
    if (controller) {
      controller.abort();
      setController(null);
    }
    setLoading(false);
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
