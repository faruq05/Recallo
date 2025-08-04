import React from "react";
import { Link } from "react-router-dom";
import Header from "../components/header";
// import aivis from "../assets/ai-assistant.png";
import RecalloVisual3D from "../components/RecalloVisual3D";
const Home = () => {
  return (
    <div className="home-container">
      <Header />
      <div className="container">
        <div className="hero">
          <div className="row">
            <div className="col-md-12">
              <div className="hero_content text-center">
                <div className="d-flex justify-content-center align-items-center mb-3">
                  <RecalloVisual3D />
                </div>
                {/* <img src={aivis} alt="ai_visualiser" className="img-fluid visual_img"/> */}
                <h1 className="grad_text">
                  Recallo: Your AI Study Partner for Smarter Retention
                </h1>
                <p className="pt-4 pb-4">
                  Recallo is an AI-driven learning companion that enhances
                  memory retention through personalized spaced repetition and
                  intelligent recall strategies perfect for students,
                  professionals, and lifelong learners.
                </p>
                <div className="hero_button">
                  <Link to="/chat" className="btn btn-cs me-3">
                    Go to Recallo
                  </Link>
                  <Link to="/signin" className="btn btn-cs me-3">
                    Sign Up Today
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* about section */}
      <div className="container">
        <div className="row">
          <div className="col-md-6">
            <div className="about_section">
              <h2 className="grad_text">About Recallo</h2>
              <p>
                Recallo is an intelligent, AI-powered learning companion
                designed to revolutionize how you retain knowledge. Leveraging
                advanced spaced repetition and adaptive memory techniques,
                Recallo helps students and lifelong learners identify what
                topics they struggle with and provides personalized, interactive
                reminders — whether through quizzes, visuals, or bite-sized
                notes. Say goodbye to cramming and guesswork; Recallo makes your
                study sessions smarter, more focused, and less stressful by
                ensuring you review the right material at the right time.
              </p>
            </div>
          </div>
          <div className="col-md-6"></div>
        </div>
        <div className="row">
          <div className="col-md-6"></div>
          <div className="col-md-6">
            <div className="about_section pt-5">
              <h2 className="grad_text">Why Recallo?</h2>
              <p>
                Traditional learning methods often leave students overwhelmed
                with what to study and when. Recallo bridges this gap by
                combining science-backed learning strategies with intuitive
                technology. Whether you're a student battling exam stress or a
                professional refreshing skills, Recallo’s smart reminders and
                engaging interface turn learning into a streamlined, efficient
                journey.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
