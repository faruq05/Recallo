import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import Chat from "./pages/Chat";
import Signin from "./pages/Signin";
import ProtectedRoute from "./components/ProtectedRoute";
import Profile from "./pages/Profile";
import Todo from "./pages/Todo";
import Topics from "./pages/Topics";
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/signin" element={<Signin />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/todo" element={<Todo />} />
        <Route path="/topics" element={<Topics />} />
      </Routes>
    </Router>
  );
}

export default App;
