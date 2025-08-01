import React, { useState, useEffect } from "react";
import { Modal, Button } from "react-bootstrap";
import { Loader2, ChevronLeft, ChevronRight } from "lucide-react";

const FlashcardViewer = ({ topic, userId, show, onHide }) => {
  const [flashcards, setFlashcards] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchFlashcards = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        "http://localhost:5000/generate_flashcards",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            topic_id: topic?.topic_id,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok || data.status !== "success") {
        // Handle specific backend error messages
        if (data.message.includes("No questions found")) {
          throw new Error(
            "This topic has no questions yet. Please create some first."
          );
        }
        if (data.message.includes("No quiz answers found")) {
          throw new Error("No quiz attempts found. Try taking a quiz first.");
        }
        throw new Error(data.message || "Failed to generate flashcards");
      }

      if (!data.data?.length) {
        throw new Error(
          "No flashcards were generated. Try taking a quiz first."
        );
      }

      setFlashcards(data.data);
      setCurrentIndex(0);
    } catch (err) {
      console.error("Flashcard error:", err);
      setError(err.message);
      setFlashcards([
        {
          concept: "Unable to Load Flashcards",
          definition: err.message,
          mistake: err.message.includes("create some")
            ? "Ask your instructor to add questions"
            : "Please try again later",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (show && topic) {
      fetchFlashcards();
    } else {
      setFlashcards([]);
      setCurrentIndex(0);
      setError(null);
    }
  }, [show, topic]);

  const handleNext = () => {
    setCurrentIndex((prev) => (prev + 1) % flashcards.length);
  };

  const handlePrev = () => {
    setCurrentIndex(
      (prev) => (prev - 1 + flashcards.length) % flashcards.length
    );
  };

  const currentCard = flashcards[currentIndex] || {
    concept: "Loading...",
    definition: "",
    mistake: "",
  };

  return (
    <Modal
      show={show}
      onHide={onHide}
      size="lg"
      centered
      backdrop="static"
      className="flashcard-modal"
    >
      <Modal.Header closeButton className="bg-dark text-white">
        <Modal.Title>
          {topic?.title || "Flashcards"}
          {isLoading && (
            <span className="ms-2">
              <Loader2 className="spin" size={18} />
            </span>
          )}
        </Modal.Title>
      </Modal.Header>

      <Modal.Body className="bg-dark text-white p-4">
        {error ? (
          <div className="alert alert-danger">
            <p>{error}</p>
            <Button
              variant="outline-light"
              onClick={fetchFlashcards}
              disabled={isLoading}
            >
              Retry
            </Button>
          </div>
        ) : (
          <div className="flashcard-container">
            <div className="flashcard-content bg-dark-800 p-4 rounded border border-secondary">
              <h4 className="gradient-text mb-3">{currentCard.concept}</h4>

              <div className="theory-content mt-3">
                <p>{currentCard.definition}</p>
                {currentCard.mistake && (
                  <div className="mistake-note mt-3 p-2 bg-dark-700 rounded">
                    <small className="text-warning">
                      {currentCard.mistake}
                    </small>
                  </div>
                )}
              </div>
            </div>

            {flashcards.length > 1 && (
              <div className="d-flex justify-content-between mt-4">
                <Button
                  variant="outline-secondary"
                  onClick={handlePrev}
                  disabled={isLoading}
                >
                  <ChevronLeft size={18} className="me-1" />
                  Previous
                </Button>

                <span className="align-self-center">
                  {currentIndex + 1} / {flashcards.length}
                </span>

                <Button
                  variant="outline-primary"
                  onClick={handleNext}
                  disabled={isLoading}
                >
                  Next
                  <ChevronRight size={18} className="ms-1" />
                </Button>
              </div>
            )}
          </div>
        )}
      </Modal.Body>

      <Modal.Footer className="bg-dark">
        <Button variant="outline-light" onClick={onHide}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default FlashcardViewer;
