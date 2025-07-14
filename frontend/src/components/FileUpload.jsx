import React from "react";
import { useEffect } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlus } from "@fortawesome/free-solid-svg-icons";
import { Tooltip } from "bootstrap";

const FileUpload = ({ onFileSelect }) => {
  useEffect(() => {
    // Activate all tooltips on the page
    const tooltipTriggerList = document.querySelectorAll(
      '[data-bs-toggle="tooltip"]'
    );
    tooltipTriggerList.forEach((tooltipTriggerEl) => {
      new Tooltip(tooltipTriggerEl);
    });
  }, []);
  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB size limit
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      alert("File size exceeds the 5MB limit.");
      return;
    }

    // Optional: Notify UI that a file was picked
    if (onFileSelect) {
      onFileSelect(file);
    }

    // Retrieve the user_id from localStorage (or any other method)
    const userId = localStorage.getItem("userId"); // Assuming user_id is saved in localStorage

    // Check if userId is available
    if (!userId) {
      console.error("User ID is not available.");
      return;
    }

    // 🔥 Send to Flask backend
    // Create FormData to send the file and user_id
    const formData = new FormData();
    formData.append("file", file); // Append the file
    formData.append("user_id", userId); // Append the user_id to the FormData
    formData.append("message", "File upload with user_id!"); // Example additional field

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
      <label htmlFor="file-upload" className="upload-icon chat_ic" data-bs-toggle="tooltip"
          data-bs-placement="top"
          title="Allowed formats: PDF, DOC, DOCX, TXT. Max size: 5 MB.">
        <FontAwesomeIcon
          icon={faPlus}
          style={{ color: "#ffffff", cursor: "pointer" }}
        />
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
