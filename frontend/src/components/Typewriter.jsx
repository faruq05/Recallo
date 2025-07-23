import React, { useEffect, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";

const Typewriter = ({ text, stop }) => {
  const [displayed, setDisplayed] = useState("");
  const intervalRef = useRef(null);

  useEffect(() => {
    let i = 0;
    setDisplayed("");

    if (stop) {
      setDisplayed(text);
      return;
    }

    intervalRef.current = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(intervalRef.current);
      }
    }, 10);

    return () => clearInterval(intervalRef.current);
  }, [text, stop]);

  return <ReactMarkdown>{displayed}</ReactMarkdown>;
};

export default Typewriter;