import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

const Typewriter = ({ text }) => {
  const [displayed, setDisplayed] = useState("");

  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      setDisplayed((prev) => prev + text.charAt(i));
      i++;
      if (i >= text.length) clearInterval(interval);
    }, 30); // Typing speed (ms per character)

    return () => clearInterval(interval); // Cleanup on unmount
  }, [text]);

  return <ReactMarkdown>{displayed}</ReactMarkdown>;
};

export default Typewriter;
