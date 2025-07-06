import React from "react";
import recalloLogo from "../assets/recallo.png";
import { EqualApproximately } from 'lucide-react';
const Header = () => {
  return (
    <header className="header">
      <nav className="navbar navbar-expand-lg">
        <div className="container-fluid">
          <a className="navbar-brand" href="/">
            <img src={recalloLogo} alt="recallo_logo" className='img-fluid logo' />
          </a>
          <button
            className="navbar-toggler ms-auto"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarSupportedContent"
            aria-controls="navbarSupportedContent"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-menu">
              <EqualApproximately />
            </span>
          </button>

          <div className="collapse navbar-collapse" id="navbarSupportedContent">
            <ul className="navbar-nav ms-auto me-auto mb-2 mb-lg-0">
              <li className="nav-item">
                <a className="nav-link active" aria-current="page" href="/">
                  Home
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="#about">
                  About
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="#features">
                  Features
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="#developers">
                  Developers
                </a>
              </li>
            </ul>
          </div>
          <div>
            <a href="/signin" className="btn btn-cs header-btn">Get Started</a>
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;
