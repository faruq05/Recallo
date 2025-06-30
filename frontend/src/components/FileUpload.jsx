import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlus } from "@fortawesome/free-solid-svg-icons";

const FileUpload = ({ onFileSelect }) => {
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Optional: Notify UI that a file was picked
    if (onFileSelect) {
      onFileSelect(file);
    }

    // 🔥 Send to Flask backend
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        console.log("✅ Upload success:", data);
      } else {
        console.error("❌ Upload failed:", data.error);
        alert("Upload failed: " + data.error);
      }
    } catch (err) {
      console.error("❌ Upload error:", err);
      alert("Upload error");
    }
  };

  return (
    <>
      <label
        htmlFor="file-upload"
        className="upload-icon chat_ic"
        title="Upload file"
      >
        <FontAwesomeIcon icon={faPlus} style={{ color: "#ffffff" }} />
      </label>
      <input
        type="file"
        id="file-upload"
        style={{ display: "none" }}
        accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,.webp"
        onChange={handleFileChange}
      />
    </>
  );
};

export default FileUpload;