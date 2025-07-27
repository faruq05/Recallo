import React, { useState, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import History from "../components/History";
import useSession from "../utils/useSession";
import {
  EqualApproximately,
  Eye,
  BookMarked,
  Clock,
  BarChart2,
  TrendingUp,
  TrendingDown,
  ChevronUp,
  ChevronDown,
  Pencil,
  Trash2,
  PackageSearch,
} from "lucide-react";
import "bootstrap/dist/css/bootstrap.min.css";
import supabase from "../utils/supabaseClient";
import ProgressBar from "react-bootstrap/ProgressBar";
import { useNavigate } from "react-router-dom";

const StudyMetrics = () => {
  const {
    user,
    userId,
    isLoggedIn,
    isSidebarOpen,
    isHistoryOpen,
    toggleSidebar,
    toggleHistory,
  } = useSession();

  const navigate = useNavigate();
  const [weakTopics, setWeakTopics] = useState([]);
  const [strongTopics, setStrongTopics] = useState([]);
  const [metrics, setMetrics] = useState({
    quizCount: 0,
    averageScore: 0,
    weakCount: 0,
    completedCount: 0,
    lastActivity: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editingTopicId, setEditingTopicId] = useState(null);
  const [newTitle, setNewTitle] = useState("");
  const [expandedSummaries, setExpandedSummaries] = useState({});

  const toggleSummary = (topicId) => {
    setExpandedSummaries((prev) => ({
      ...prev,
      [topicId]: !prev[topicId],
    }));
  };

  useEffect(() => {
    const fetchStudyMetrics = async () => {
      try {
        setLoading(true);
        setError(null);

        if (!userId) {
          setLoading(false);
          return;
        }

        // Fetch all quiz attempts
        const { data: attempts, error: attemptsError } = await supabase
          .from("quiz_attempts")
          .select("attempt_id, topic_id, score, submitted_at")
          .eq("user_id", userId)
          .order("submitted_at", { ascending: false });

        if (attemptsError) throw attemptsError;

        // Fetch topics with their latest attempt
        const { data: topics, error: topicsError } = await supabase
          .from("topics")
          .select(
            "topic_id, title, file_name, created_at, topic_summary, topic_status"
          )
          .eq("user_id", userId)
          .eq("archive_status", "not_archived");

        if (topicsError) throw topicsError;

        // Process metrics
        const quizCount = attempts.length;
        const totalScore = attempts.reduce(
          (sum, attempt) => sum + (attempt.score || 0),
          0
        );
        const averageScore =
          quizCount > 0 ? (totalScore / quizCount).toFixed(1) : 0;

        // Find latest attempt for each topic
        const topicMetrics = topics.map((topic) => {
          const topicAttempts = attempts.filter(
            (a) => a.topic_id === topic.topic_id
          );
          const latestAttempt = topicAttempts[0];
          return {
            ...topic,
            latestScore: latestAttempt?.score || null,
            lastAttemptDate: latestAttempt?.submitted_at || null,
            attemptCount: topicAttempts.length,
          };
        });

        // Separate weak and strong topics
        const weak = topicMetrics.filter(
          (topic) => topic.latestScore !== null && topic.latestScore <= 7
        );
        const strong = topicMetrics.filter(
          (topic) => topic.latestScore !== null && topic.latestScore > 7
        );

        setWeakTopics(weak);
        setStrongTopics(strong);
        setMetrics({
          quizCount,
          averageScore,
          weakCount: weak.length,
          completedCount: strong.length,
          lastActivity: attempts[0]?.submitted_at || null,
        });
      } catch (err) {
        console.error("Error fetching study metrics:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchStudyMetrics();
  }, [userId]);

  const handleEditTopic = async (topicId) => {
    if (!newTitle.trim()) return;

    const { error } = await supabase
      .from("topics")
      .update({ title: newTitle })
      .eq("topic_id", topicId);

    if (error) {
      alert("Failed to update topic");
      console.error(error);
    } else {
      setEditingTopicId(null);
      setNewTitle("");
      // Refresh data
      const updatedWeak = weakTopics.map((topic) =>
        topic.topic_id === topicId ? { ...topic, title: newTitle } : topic
      );
      const updatedStrong = strongTopics.map((topic) =>
        topic.topic_id === topicId ? { ...topic, title: newTitle } : topic
      );
      setWeakTopics(updatedWeak);
      setStrongTopics(updatedStrong);
    }
  };

  const handleDeleteTopic = async (topicId) => {
    const confirmDelete = confirm(
      "Are you sure you want to delete this topic?"
    );
    if (!confirmDelete) return;

    const { error } = await supabase
      .from("topics")
      .delete()
      .eq("topic_id", topicId);

    if (error) {
      alert("Failed to delete topic");
      console.error(error);
    } else {
      setWeakTopics((prev) =>
        prev.filter((topic) => topic.topic_id !== topicId)
      );
      setStrongTopics((prev) =>
        prev.filter((topic) => topic.topic_id !== topicId)
      );
    }
  };

  const handleTakeExam = (topic) => {
    navigate("/exam", {
      state: {
        topicId: topic.topic_id,
        topicTitle: topic.title,
        fileName: topic.file_name,
      },
    });
  };

  if (!isLoggedIn) {
    return (
      <div className="chat chat-wrapper d-flex min-vh-100">
        <Sidebar
          isOpen={isSidebarOpen}
          toggleSidebar={toggleSidebar}
          toggleHistory={toggleHistory}
          isHistoryOpen={isHistoryOpen}
          isLoggedIn={isLoggedIn}
        />
        <div className="chat-content flex-grow-1 p-4 text-white">
          <div className="container text-center mt-5">
            <h3>Please sign in to view study metrics</h3>
          </div>
        </div>
      </div>
    );
  }

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
          userId={userId}
          isHistoryOpen={isHistoryOpen}
          onClose={toggleHistory}
        />
      </div>

      <div className="chat-content flex-grow-1 p-4 text-white">
        <div className="container text-center mb-4 mt-4">
          <h2 className="grad_text">Study Metrics</h2>
        </div>

        {/* Metrics Overview */}
        <div className="container mb-5">
          <div className="row">
            <div className="col-md-6 col-lg-4 mb-3">
              <div className="metric-card bg-dark p-3 rounded h-100">
                <div className="d-flex align-items-center">
                  <BarChart2 size={24} className="me-2 text-primary" />
                  <h5 className="mb-0">Quizzes Taken</h5>
                </div>
                <h4 className="mt-2">{metrics.quizCount}</h4>
              </div>
            </div>
            <div className="col-md-6 col-lg-4 mb-3">
              <div className="metric-card bg-dark p-3 rounded h-100">
                <div className="d-flex align-items-center">
                  <TrendingUp size={24} className="me-2 text-success" />
                  <h5 className="mb-0">Average Score</h5>
                </div>
                <h4 className="mt-2">{metrics.averageScore}/10</h4>
                <ProgressBar
                  now={metrics.averageScore * 10}
                  variant="success"
                  className="mt-2"
                />
              </div>
            </div>
            <div className="col-md-6 col-lg-4 mb-3">
              <div className="metric-card bg-dark p-3 rounded h-100">
                <div className="d-flex align-items-center">
                  <Clock size={24} className="me-2 text-info" />
                  <h5 className="mb-0">Last Activity</h5>
                </div>
                <h4 className="mt-2">
                  {metrics.lastActivity
                    ? new Date(metrics.lastActivity).toLocaleDateString()
                    : "No activity yet"}
                </h4>
              </div>
            </div>
            <div className="col-md-6 col-lg-4 mb-3">
              <div className="metric-card bg-dark p-3 rounded h-100">
                <div className="d-flex align-items-center">
                  <TrendingDown size={24} className="me-2 text-danger" />
                  <h5 className="mb-0">Weak Topics</h5>
                </div>
                <h4 className="mt-2">{metrics.weakCount}</h4>
              </div>
            </div>
            <div className="col-md-6 col-lg-4 mb-3">
              <div className="metric-card bg-dark p-3 rounded h-100">
                <div className="d-flex align-items-center">
                  <TrendingUp size={24} className="me-2 text-success" />
                  <h5 className="mb-0">Strong Topics</h5>
                </div>
                <h4 className="mt-2">{metrics.completedCount}</h4>
              </div>
            </div>
          </div>
        </div>

        {/* Weak Topics Section */}
        <div className="container mb-5">
          <h4 className="grad-text mb-4">
            <TrendingDown className="me-2 text-danger" />
            <span className="grad_text">Weak Areas</span>
          </h4>

          {loading ? (
            <div className="text-center">
              <div className="spinner-border text-light" role="status" />
            </div>
          ) : weakTopics.length === 0 ? (
            <div className="alert alert-metrics">
              No weak topics found. Keep up the good work!
            </div>
          ) : (
            // Group topics by file_name
            Object.entries(
              weakTopics.reduce((acc, topic) => {
                const fileName = topic.file_name || "Uncategorized";
                if (!acc[fileName]) {
                  acc[fileName] = [];
                }
                acc[fileName].push(topic);
                return acc;
              }, {})
            ).map(([fileName, topics]) => (
              <div key={fileName} className="mb-4">
                {/* File name header - now only once per group */}
                <div className="d-flex align-items-center mb-2">
                  <PackageSearch size={30} className="me-2" />
                  <h4 className="text-white">{fileName}</h4>
                </div>

                <div className="row">
                  {topics.map((topic) => (
                    <div
                      className="col-sm-6 col-md-6 col-xl-4 mb-4"
                      key={topic.topic_id}
                    >
                      <div className="card topic_card text-white">
                        <div className="card-body">
                          <div className="topic_status d-flex justify-content-between align-items-center mb-3">
                            <span className="badge badge-weak">Weak</span>
                            <p className="card-text d-flex align-items-center small text-muted">
                              <Clock size={14} className="me-1" />
                              {new Date(
                                topic.lastAttemptDate || topic.created_at
                              ).toLocaleDateString()}
                            </p>
                          </div>

                          <div className="d-flex justify-content-between align-items-start mb-2">
                            {editingTopicId === topic.topic_id ? (
                              <input
                                type="text"
                                className="form-control form-control-sm me-2"
                                value={newTitle}
                                onChange={(e) => setNewTitle(e.target.value)}
                                onBlur={() => handleEditTopic(topic.topic_id)}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter")
                                    handleEditTopic(topic.topic_id);
                                  if (e.key === "Escape") {
                                    setEditingTopicId(null);
                                    setNewTitle("");
                                  }
                                }}
                                autoFocus
                              />
                            ) : (
                              <h5 className="card-title mb-0">{topic.title}</h5>
                            )}
                            <div className="d-flex gap-2">
                              <Pencil
                                className="edit-icon"
                                size={18}
                                style={{ cursor: "pointer" }}
                                onClick={() => {
                                  setEditingTopicId(topic.topic_id);
                                  setNewTitle(topic.title);
                                }}
                              />
                              <Trash2
                                className="edit-icon"
                                size={18}
                                style={{ cursor: "pointer" }}
                                onClick={() =>
                                  handleDeleteTopic(topic.topic_id)
                                }
                              />
                            </div>
                          </div>

                          <div className="d-flex justify-content-between align-items-center mb-2">
                            <span className="text-danger fw-bold">
                              {topic.latestScore}/10
                            </span>
                            <span className="small text-muted">
                              {topic.attemptCount} attempt
                              {topic.attemptCount !== 1 ? "s" : ""}
                            </span>
                          </div>

                          <button
                            className="btn btn-sm btn-outline w-100 p-0 d-flex justify-content-between align-items-center text-white mt-2"
                            onClick={() => toggleSummary(topic.topic_id)}
                          >
                            Topic Summary
                            {expandedSummaries[topic.topic_id] ? (
                              <ChevronUp size={16} />
                            ) : (
                              <ChevronDown size={16} />
                            )}
                          </button>

                          {expandedSummaries[topic.topic_id] && (
                            <div className="card-topic-summary text-white small mt-2">
                              {topic.topic_summary || "No summary available."}
                            </div>
                          )}

                          <div className="d-flex justify-content-between mt-3">
                            <button className="btn btn-sm btn-outline-light">
                              <Eye size={16} className="me-1" />
                              Flashcards
                            </button>
                            <button
                              className="btn btn-sm btn-answer"
                              onClick={() => handleTakeExam(topic)}
                            >
                              Take Exam
                            </button>
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

        {/* Strong Topics Section */}
        <div className="container">
          <h4 className="grad-text mb-4">
            <TrendingUp className="me-2 text-success" />
            <span className="grad_text">Strong Areas</span>
          </h4>

          {loading ? (
            <div className="text-center">
              <div className="spinner-border text-light" role="status" />
            </div>
          ) : strongTopics.length === 0 ? (
            <div className="alert alert-metrics">
              No strong topics yet. Keep practicing!
            </div>
          ) : (
            // Group topics by file_name
            Object.entries(
              strongTopics.reduce((acc, topic) => {
                const fileName = topic.file_name || "Uncategorized";
                if (!acc[fileName]) {
                  acc[fileName] = [];
                }
                acc[fileName].push(topic);
                return acc;
              }, {})
            ).map(([fileName, topics]) => (
              <div key={fileName} className="mb-4">
                {/* File name header - now only once per group */}
                <div className="d-flex align-items-center mb-2">
                  <PackageSearch size={16} className="me-2 text-muted" />
                  <small className="text-muted text-truncate">{fileName}</small>
                </div>

                <div className="row">
                  {topics.map((topic) => (
                    <div
                      className="col-sm-6 col-md-6 col-xl-4 mb-4"
                      key={topic.topic_id}
                    >
                      <div className="card topic_card text-white">
                        <div className="card-body">
                          <div className="topic_status d-flex justify-content-between align-items-center mb-3">
                            <span className="badge badge-completed">
                              Strong
                            </span>
                            <p className="card-text d-flex align-items-center small text-muted">
                              <Clock size={14} className="me-1" />
                              {new Date(
                                topic.lastAttemptDate || topic.created_at
                              ).toLocaleDateString()}
                            </p>
                          </div>

                          <div className="d-flex justify-content-between align-items-start mb-2">
                            {editingTopicId === topic.topic_id ? (
                              <input
                                type="text"
                                className="form-control form-control-sm me-2"
                                value={newTitle}
                                onChange={(e) => setNewTitle(e.target.value)}
                                onBlur={() => handleEditTopic(topic.topic_id)}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter")
                                    handleEditTopic(topic.topic_id);
                                  if (e.key === "Escape") {
                                    setEditingTopicId(null);
                                    setNewTitle("");
                                  }
                                }}
                                autoFocus
                              />
                            ) : (
                              <h5 className="card-title mb-0">{topic.title}</h5>
                            )}
                            <div className="d-flex gap-2">
                              <Pencil
                                className="edit-icon"
                                size={18}
                                style={{ cursor: "pointer" }}
                                onClick={() => {
                                  setEditingTopicId(topic.topic_id);
                                  setNewTitle(topic.title);
                                }}
                              />
                              <Trash2
                                className="edit-icon"
                                size={18}
                                style={{ cursor: "pointer" }}
                                onClick={() =>
                                  handleDeleteTopic(topic.topic_id)
                                }
                              />
                            </div>
                          </div>

                          <div className="d-flex justify-content-between align-items-center mb-2">
                            <span className="text-success fw-bold">
                              {topic.latestScore}/10
                            </span>
                            <span className="small text-muted">
                              {topic.attemptCount} attempt
                              {topic.attemptCount !== 1 ? "s" : ""}
                            </span>
                          </div>

                          <button
                            className="btn btn-sm btn-outline w-100 p-0 d-flex justify-content-between align-items-center text-white mt-2"
                            onClick={() => toggleSummary(topic.topic_id)}
                          >
                            Topic Summary
                            {expandedSummaries[topic.topic_id] ? (
                              <ChevronUp size={16} />
                            ) : (
                              <ChevronDown size={16} />
                            )}
                          </button>

                          {expandedSummaries[topic.topic_id] && (
                            <div className="card-topic-summary text-white small mt-2">
                              {topic.topic_summary || "No summary available."}
                            </div>
                          )}

                          <button className="btn btn-sm btn-outline-light w-100 mt-3">
                            <Eye size={16} className="me-1" />
                            Flashcards
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>

        <EqualApproximately
          className="d-md-none position-fixed top-0 start-0 m-3"
          onClick={toggleSidebar}
          style={{ zIndex: 99 }}
        />
      </div>
    </div>
  );
};

export default StudyMetrics;
