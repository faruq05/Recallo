import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

const Typewriter = ({ text }) => {
  const [displayed, setDisplayed] = useState("");

  useEffect(() => {
    let i = 0;
    setDisplayed("");

    const interval = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) clearInterval(interval);
    }, 10);

    return () => clearInterval(interval);
  }, [text]);

  return <ReactMarkdown>{displayed}</ReactMarkdown>;
};

export default Typewriter;