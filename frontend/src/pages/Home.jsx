import React from "react";
import { Link } from "react-router-dom";
import Header from "../components/header";
import aivis from "../assets/ai-assistant.png";
const Home = () => {
  return (
    <div className="home-container">
      <Header />
      <div className="container">
        <div className="hero">
          <div className="row">
            <div className="col-md-12">
              <div className="hero_content text-center">
                <img src={aivis} alt="ai_visualiser" className="img-fluid visual_img"/>
                <h1 className="grad_text">Recallo: Your AI Study Partner for Smarter Retention</h1>
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
    </div>
  );
};

export default Home;
