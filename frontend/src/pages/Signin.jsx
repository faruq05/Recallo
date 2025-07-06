import "../index.css";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { createClient } from "@supabase/supabase-js";
import { Auth } from "@supabase/auth-ui-react";
import { ThemeSupa } from "@supabase/auth-ui-shared";
import Header from "../components/header";

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_KEY
);

export default function SignIn() {
  const [session, setSession] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      if (session) {
        localStorage.setItem("userToken", session.access_token);
        navigate("/chat");
      }
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
       if (session) {
        localStorage.setItem("userToken", session.access_token);
        navigate("/chat");
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const signout = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) console.error("Error signing out:", error.message);
  };

  if (!session) {
    return (
      <div className="home-container">
        <Header />
        <div className="container hero">
          <div className="row">
            <div className="col-md-4">
              <div className="p-4 shadow rounded bg-white auth-card">
                <Auth
                  supabaseClient={supabase}
                  appearance={{ theme: ThemeSupa }}
                  providers={["google", "github"]}
                  redirectTo="http://localhost:5173/signin"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  } else {
    return (
      <div className="home-container">
        <div className="container hero">
          <div className="row">
            <div className="col-md-4">
              <div className="p-4 shadow rounded bg-white">
                <p style={{ color: "#000" }}>
                  You are successfully logged in as {session.user.email}.
                  Explore the features and start your learning journey.
                </p>
                <button onClick={signout} className="btn btn-cs">
                  Sign Out
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}
