// utils/useSession.js
import { useState, useEffect } from "react";
import { createClient } from "@supabase/supabase-js";

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_KEY
);

const useSession = () => {
  const [session, setSession] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth > 767);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });

    return () => listener?.subscription.unsubscribe();
  }, []);

  useEffect(() => {
    const handleResize = () => {
      setIsSidebarOpen(window.innerWidth > 767);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);
  const toggleHistory = () => setIsHistoryOpen((prev) => !prev);

  return {
    session,
    userId: session?.user?.id || null,
    isLoggedIn: !!session,
    isSidebarOpen,
    isHistoryOpen,
    toggleSidebar,
    toggleHistory,
  };
};

export default useSession;
