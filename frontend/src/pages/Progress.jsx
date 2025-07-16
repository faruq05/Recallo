// Progress.jsx
import React, { useState } from "react";
import Sidebar from "../components/Sidebar";
import History from "../components/History";
import useSession from "../utils/useSession";
import { EqualApproximately, ChevronDown, ChevronUp } from "lucide-react";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import "bootstrap/dist/css/bootstrap.min.css";

const Progress = () => {
  const {
    user,
    isLoggedIn,
    isSidebarOpen,
    isHistoryOpen,
    toggleSidebar,
    toggleHistory,
  } = useSession();

  const [expandedTopics, setExpandedTopics] = useState({});

  const toggleHistoryView = (topicId) => {
    setExpandedTopics((prev) => ({
      ...prev,
      [topicId]: !prev[topicId],
    }));
  };

  const topicResults = [
    {
      topicId: "t1",
      fileName: "EEE 111 lab manual (1–7).pdf",
      topicTitle: "Photosynthesis",
      latestScore: 7,
      history: [
        { score: 7, timestamp: "2024-07-01 14:10" },
        { score: 5, timestamp: "2024-06-25 11:00" },
        { score: 6, timestamp: "2024-06-20 16:40" },
      ],
    },
    {
      topicId: "t2",
      fileName: "EEE 111 lab manual (1–7).pdf",
      topicTitle: "Cell Division",
      latestScore: 8,
      history: [
        { score: 6, timestamp: "2024-06-22 13:30" },
        { score: 8, timestamp: "2024-06-30 09:15" },
      ],
    },
  ];

  const groupedTopics = topicResults.reduce((acc, curr) => {
    if (curr.latestScore === null) return acc;
    if (!acc[curr.fileName]) acc[curr.fileName] = [];
    acc[curr.fileName].push(curr);
    return acc;
  }, {});

  //   if no exam given print a message
  const hasResults = Object.keys(groupedTopics).length > 0;
  
  return (
    <div className="chat chat-wrapper d-flex min-vh-100">
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
          userId={user?.id}
          isHistoryOpen={isHistoryOpen}
          onClose={toggleHistory}
        />
      </div>

      <div className="chat-content flex-grow-1 p-4 text-white">
        <div className="container text-center mb-4 mt-4">
          <h2 className="grad_text">Upload Your Files</h2>
        </div>

        <div className="container">
          {!hasResults ? (
            <div className="text-center mt-5">
              <h5 className="text-warning">
                📌 Take exams for progress overview
              </h5>
            </div>
          ) : (
            Object.entries(groupedTopics).map(([fileName, topics]) => (
              <div key={fileName} className="mb-5">
                <h4 className="text-white mb-3">
                  <i className="bi bi-file-earmark-text me-2"></i>
                  {fileName}
                </h4>
                <div className="row">
                  {topics.map((topic, idx) => (
                    <div className="col-md-6 col-xl-4 mb-4" key={idx}>
                      <div className="card topic_card text-white">
                        <div className="card-body">
                          <h5 className="card-title mb-1">
                            {topic.topicTitle}
                          </h5>
                          <div className="my-3" style={{ width: 60 }}>
                            <CircularProgressbar
                              value={topic.latestScore * 10}
                              text={`${topic.latestScore}/10`}
                              styles={buildStyles({
                                textColor: "white",
                                pathColor:
                                  topic.latestScore >= 8
                                    ? "#28a745"
                                    : "#ffc107",
                                trailColor: "#444",
                              })}
                            />
                          </div>

                          <div className="mt-3 marks_progress">
                            <button
                              className="btn btn-sm btn-outline w-100 d-flex justify-content-between align-items-center text-white ans_drp"
                              onClick={() => toggleHistoryView(topic.topicId)}
                            >
                              Exam Marks History
                              {expandedTopics[topic.topicId] ? (
                                <ChevronUp size={16} />
                              ) : (
                                <ChevronDown size={16} />
                              )}
                            </button>
                            {expandedTopics[topic.topicId] && (
                              <div className="mt-2">
                                {topic.history.length > 0 &&
                                  topic.history.map((entry, i) => {
                                    const prev = topic.history[i - 1]?.score;
                                    const delta =
                                      prev !== undefined
                                        ? entry.score - prev
                                        : 0;
                                    const trend =
                                      delta > 0 ? (
                                        <span className="text-success">
                                          +{delta} ↑
                                        </span>
                                      ) : delta < 0 ? (
                                        <span className="text-danger">
                                          {delta} ↓
                                        </span>
                                      ) : (
                                        <span className="text-secondary">
                                          0 →
                                        </span>
                                      );
                                    return (
                                      <div key={i} className="marks_hitory">
                                        <div className="d-flex justify-content-between test_details_progress">
                                          <strong>Test {i + 1}</strong>
                                          <span className="text-muted small">
                                            {entry.timestamp}
                                          </span>
                                        </div>
                                        <div className="d-flex justify-content-between align-items-center mt-2">
                                          <span>
                                            <strong>{entry.score}/10</strong> (
                                            {trend})
                                          </span>
                                          <button
                                            className="btn btn-sm btn-answer"
                                            onClick={() =>
                                              alert(
                                                `Redirect to review of Test ${
                                                  i + 1
                                                }`
                                              )
                                            }
                                          >
                                            Answer Analysis
                                          </button>
                                        </div>
                                      </div>
                                    );
                                  })}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>

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

export default Progress;
