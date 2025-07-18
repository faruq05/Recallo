// Topics.jsx
import React, { useState, useEffect } from "react";
import FileUpload from "../components/FileUpload";
import Sidebar from "../components/Sidebar";
import History from "../components/History";
import useSession from "../utils/useSession";
import { EqualApproximately, Trash2, Pencil } from "lucide-react";
import ProgressBar from "react-bootstrap/ProgressBar";
import "bootstrap/dist/css/bootstrap.min.css";
import { createClient } from "@supabase/supabase-js";
import { PackageSearch } from 'lucide-react';



const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_KEY
);

const Topics = () => {
  const {
    userId,
    user,
    isLoggedIn,
    isSidebarOpen,
    isHistoryOpen,
    toggleSidebar,
    toggleHistory,
  } = useSession();

  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [topics, setTopics] = useState([]);

  const simulateProgress = () => {
    setUploading(true);
    let val = 0;
    const interval = setInterval(() => {
      val += 5;
      setProgress(val);
      if (val >= 100) {
        clearInterval(interval);
        setUploading(false);
      }
    }, 150);
  };

  const handleFileSelect = async (file) => {
    if (!userId) {
      console.error("User ID is missing.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("user_id", userId);

    setUploadedFiles((prev) => [...prev, { file, done: false }]);
    setProgress(0);
    simulateProgress();

    try {
      const response = await fetch("http://localhost:5000/quiz-question", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      if (response.ok) {
        alert(result.message || "File processed successfully. Topics saved to Supabase.");
      } else {
        // Show backend message even on error status
        alert(result.message || result.error || "Something went wrong during upload.");
      }

      console.log("Upload result:", result);

    } catch (error) {
      console.error("File processing error:", error);
    } finally {
      setUploading(false);
    }
  };

  useEffect(() => {
    if (!uploading && uploadedFiles.length > 0) {
      setUploadedFiles((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].done = true;
        return updated;
      });

      // For real Supabase usage
      // fetchTopics();
    }
  }, [uploading]);

  // const fetchTopics = async () => {
  //   const { data, error } = await supabase
  //     .from("topics")
  //     .select("*")
  //     .eq("user_id", userId)
  //     .order("created_at", { ascending: false });

  //   if (!error) setTopics(data);
  // };

  return (
    <div className="chat chat-wrapper d-flex min-vh-100">
      {/* Sidebar and History */}
      <div className={`sidebar-area ${isSidebarOpen ? "open" : "collapsed"}`}>
        <Sidebar
          isOpen={isSidebarOpen}
          toggleSidebar={toggleSidebar}
          toggleHistory={toggleHistory}
          isHistoryOpen={isHistoryOpen}
          isLoggedIn={isLoggedIn}
        />
        <History
          isLoggedIn={isLoggedIn}
          userId={userId}
          isHistoryOpen={isHistoryOpen}
          onClose={toggleHistory}
        />
      </div>

      {/* Main Content */}
      <div className="chat-content flex-grow-1 p-4 text-white d-flex flex-column">
        {/* Header */}
        <div className="container text-center mb-4">
          <h2 className="grad_text">Upload Your PDF</h2>
        </div>

        {/* Upload Section */}
        <div className="container mb-4">
          <div className="row justify-content-center">
            <div className="col-xl-6 col-md-8">
              <div
                className="border border-secondary rounded p-5 bg-secondary bg-opacity-25 d-flex flex-column align-items-center justify-content-center"
                style={{ cursor: "pointer" }}
              >
                <FileUpload onFileSelect={handleFileSelect} />
                <p className="lead mt-3">Upload your files to generate topics</p>
              </div>
            </div>
          </div>
        </div>

        {/* Progress / Uploaded Files Section */}
        <div className="container">
          {uploading ? (
            <div className="text-center">
              <h5 className="text-info grad_text">⏳ Processing your file...</h5>
              <ProgressBar animated now={progress} label={`${progress}%`} className="mt-3" />
            </div>
          ) : (
            uploadedFiles
              .filter((f) => f.done)
              .map((entry, i) => (
                <div className="mb-5" key={i}>
                  <h4 className="text-white mb-3">
                    <PackageSearch className="me-2" />
                    {entry.file.name}
                  </h4>
                  <div className="row">
                    {["Photosynthesis", "Cell Division", "Genetics"].map((title, idx) => (
                      <div className="col-md-6 col-xl-4 mb-4" key={idx}>
                        <div className="card topic_card text-white">
                          <div className="card-body">
                            <div className="d-flex justify-content-between align-items-start mb-2">
                              <h5 className="card-title mb-0">{title}</h5>
                              <div className="d-flex gap-2">
                                <Pencil size={18} style={{ cursor: "pointer" }} />
                                <Trash2 size={18} style={{ cursor: "pointer" }} />
                              </div>
                            </div>
                            <p className="card-text">Topic auto-generated from your uploaded file.</p>
                            <button className="btn btn-sm btn-cs">Take Exam</button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))
          )}
        </div>

        {/* Sidebar Toggle Button */}
        <span className="navbar-toggler-menu">
          <EqualApproximately
            className="d-md-none position-fixed top-0 start-0 m-3"
            onClick={toggleSidebar}
            style={{ zIndex: 99 }}
          />
        </span>
      </div>
    </div>
  );

};

export default Topics;


// import React, { useState, useEffect } from "react";
// import FileUpload from "../components/FileUpload";
// import Sidebar from "../components/Sidebar";
// import History from "../components/History";
// import useSession from "../utils/useSession";
// import { EqualApproximately, Trash2, Pencil } from "lucide-react";
// import ProgressBar from "react-bootstrap/ProgressBar";
// import "bootstrap/dist/css/bootstrap.min.css";
// import { createClient } from "@supabase/supabase-js";
// import { PackageSearch } from 'lucide-react';

// const supabase = createClient(
//   import.meta.env.VITE_SUPABASE_URL,
//   import.meta.env.VITE_SUPABASE_KEY
// );

// const Topics = () => {
//   const {
//     userId,
//     isSidebarOpen,
//     isHistoryOpen,
//     toggleSidebar,
//     toggleHistory,
//   } = useSession();

//   const [uploadedFiles, setUploadedFiles] = useState([]);
//   const [progress, setProgress] = useState(0);
//   const [uploading, setUploading] = useState(false);

//   const simulateProgress = () => {
//     setUploading(true);
//     let val = 0;
//     const interval = setInterval(() => {
//       val += 5;
//       setProgress(val);
//       if (val >= 100) {
//         clearInterval(interval);
//         setUploading(false);
//       }
//     }, 150);
//   };

//   useEffect(() => {
//     if (!uploading && uploadedFiles.length > 0) {
//       setUploadedFiles((prev) => {
//         const updated = [...prev];
//         updated[updated.length - 1].done = true;
//         return updated;
//       });
//     }
//   }, [uploading]);

//   return (
//     <div className="chat chat-wrapper d-flex min-vh-100">
//       <div className={`sidebar-area ${isSidebarOpen ? "open" : "collapsed"}`}>
//         <Sidebar
//           isOpen={isSidebarOpen}
//           toggleSidebar={toggleSidebar}
//           toggleHistory={toggleHistory}
//           isHistoryOpen={isHistoryOpen}
//           isLoggedIn={userId}
//         />
//         <History
//           isLoggedIn={userId}
//           userId={userId}
//           isHistoryOpen={isHistoryOpen}
//           onClose={toggleHistory}
//         />
//       </div>

//       <div className="chat-content flex-grow-1 p-4 text-white d-flex flex-column">
//         <div className="container text-center mb-4">
//           <h2 className="grad_text">Upload Your PDF</h2>
//         </div>

//         <div className="container mb-4">
//           <div className="row justify-content-center">
//             <div className="col-xl-6 col-md-8">
//               <div
//                 className="border border-secondary rounded p-5 bg-secondary bg-opacity-25 d-flex flex-column align-items-center justify-content-center"
//               >
//                 <FileUpload uploadTarget="/quiz-question" inputId="quiz-upload" />
//                 <p className="lead mt-3">Upload your files to generate topics</p>

//                 <FileUpload uploadTarget="/upload" inputId="general-upload" />
//                 <p className="lead mt-3">Upload general files (notes, docs)</p>
//               </div>
//             </div>
//           </div>
//         </div>

//         <div className="container">
//           {uploading ? (
//             <div className="text-center">
//               <h5 className="text-info grad_text">⏳ Processing your file...</h5>
//               <ProgressBar animated now={progress} label={`${progress}%`} className="mt-3" />
//             </div>
//           ) : (
//             uploadedFiles
//               .filter((f) => f.done)
//               .map((entry, i) => (
//                 <div className="mb-5" key={i}>
//                   <h4 className="text-white mb-3">
//                     <PackageSearch className="me-2" />
//                     {entry.file.name}
//                   </h4>
//                   <div className="row">
//                     {["Photosynthesis", "Cell Division", "Genetics"].map((title, idx) => (
//                       <div className="col-md-6 col-xl-4 mb-4" key={idx}>
//                         <div className="card topic_card text-white">
//                           <div className="card-body">
//                             <div className="d-flex justify-content-between align-items-start mb-2">
//                               <h5 className="card-title mb-0">{title}</h5>
//                               <div className="d-flex gap-2">
//                                 <Pencil size={18} style={{ cursor: "pointer" }} />
//                                 <Trash2 size={18} style={{ cursor: "pointer" }} />
//                               </div>
//                             </div>
//                             <p className="card-text">Topic auto-generated from your uploaded file.</p>
//                             <button className="btn btn-sm btn-cs">Take Exam</button>
//                           </div>
//                         </div>
//                       </div>
//                     ))}
//                   </div>
//                 </div>
//               ))
//           )}
//         </div>

//         <span className="navbar-toggler-menu">
//           <EqualApproximately
//             className="d-md-none position-fixed top-0 start-0 m-3"
//             onClick={toggleSidebar}
//             style={{ zIndex: 99 }}
//           />
//         </span>
//       </div>
//     </div>
//   );
// };

// export default Topics;
