import React, { useState, useEffect } from "react";
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
    userId,
    isLoggedIn,
    isSidebarOpen,
    isHistoryOpen,
    toggleSidebar,
    toggleHistory,
  } = useSession();

  const [expandedTopics, setExpandedTopics] = useState({});
  const [progressData, setProgressData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProgress = async () => {
      setLoading(true);
      try {
        const response = await fetch(`http://localhost:5000/api/progress/${userId}`);
        if (response.ok) {
          const data = await response.json();
          setProgressData(data);
        } else {
          console.error("Error fetching progress:", response.statusText);
          setProgressData([]);
        }
      } catch (error) {
        console.error("Error fetching progress:", error.message);
        setProgressData([]);
      }
      setLoading(false);
    };

    if (userId) {
      fetchProgress();
    }
  }, [userId]);

  const toggleHistoryView = (topicId) => {
    setExpandedTopics((prev) => ({
      ...prev,
      [topicId]: !prev[topicId],
    }));
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  // Transform the API data to match the frontend structure
 const transformData = (data) => {
  return data.map((item, index) => {
    // Get the latest attempt (last item in chronologically sorted array)
    const attemptHistory = item.attempt_history || [];
    const latestAttempt = attemptHistory[attemptHistory.length - 1] || {};

    return {
      topicId: item.topic_id || `t${index + 1}`,
      fileName: item.file_name || "Document",
      topicTitle: item.topic_title || `Topic ${index + 1}`,
      latestScore: item.latest_score || latestAttempt.score || 0, // Use backend's latest_score first
      latestAttemptNumber: latestAttempt.attempt_number || (item.total_attempts || 0),
      totalAttempts: item.total_attempts || 1,
      overallProgress: item.overall_progress_percent || 0,
      history: attemptHistory.slice().reverse() // Reverse for display (newest first)
    };
  });
};
  const transformedData = transformData(progressData);
  
  const groupedTopics = transformedData.reduce((acc, curr) => {
    if (curr.latestScore === null) return acc;
    if (!acc[curr.fileName]) acc[curr.fileName] = [];
    acc[curr.fileName].push(curr);
    return acc;
  }, {});

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
          <h2 className="grad_text">Your Progress Overview</h2>
        </div>

        <div className="container">
          {loading ? (
            <div className="d-flex justify-content-center align-items-center" style={{ height: "150px" }}>
              <div className="spinner-border text-primary" role="status" />
            </div>
          ) : !hasResults ? (
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
                              value={(topic.latestScore / 10) * 100}
                              text={`${topic.latestScore.toFixed(1)}/10`}
                              styles={buildStyles({
                                textColor: "white",
                                pathColor:
                                  topic.latestScore >= 8
                                    ? "#28a745"
                                    : topic.latestScore >= 6
                                    ? "#ffc107"
                                    : "#dc3545",
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
                                  topic.history.map((attempt, i) => {
                                    const isFirst = i === topic.history.length - 1; // Last item is first attempt
                                    const isLatest = i === 0; // First item is latest attempt
                                    const improvement = attempt.improvement !== null ? attempt.improvement : 0;

                                    return (
                                      <div key={i} className="marks_hitory">
                                        <div className="d-flex justify-content-between test_details_progress">
                                          <strong>
                                            Test {attempt.attempt_number || i + 1}
                                            
                                          </strong>
                                          <span className="text-muted small">
                                            {formatTimestamp(attempt.submitted_at)}
                                          </span>
                                        </div>
                                        <div className="d-flex justify-content-between align-items-center mt-2">
                                          <span>
                                            <strong>{attempt.score}/10</strong> (
                                            {improvement > 0 ? (
                                              <span className="text-success">
                                                +{improvement.toFixed(1)} ↑
                                              </span>
                                            ) : improvement < 0 ? (
                                              <span className="text-danger">
                                                {improvement.toFixed(1)} ↓
                                              </span>
                                            ) : (
                                              <span className="text-secondary">
                                                {isFirst ? "First attempt" : "0 →"}
                                              </span>
                                            )})
                                          </span>
                                          <button
                                            className="btn btn-sm btn-answer"
                                            onClick={() =>
                                              alert(
                                                `Redirect to review of Test ${
                                                  attempt.attempt_number || i + 1
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