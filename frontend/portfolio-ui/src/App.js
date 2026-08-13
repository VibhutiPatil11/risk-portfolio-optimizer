import { useState } from "react";
import UniverseDashboard from "./pages/UniverseDashboard";
import Dashboard from "./pages/Dashboard";
import KYCPage from "./pages/KYCPage";
import LoginPage from "./pages/LoginPage";

function App() {
  // Three possible views: "login" | "kyc" | "dashboard"
  const deriveView = () => {
    const token     = localStorage.getItem("token");
    const kycStatus = localStorage.getItem("kyc_status");
    if (!token) return "login";
    if (kycStatus !== "approved") return "kyc";
    return "dashboard";
  };

  const [view, setView] = useState(deriveView);
  const [dashboardScreen, setDashboardScreen] = useState("universe");
  const [portfolioStocks, setPortfolioStocks] = useState([]);

  // Called by LoginPage after a successful /signup response.
  // The server returns a token + kyc_status:"pending" so we go straight to KYC.
  const handleSignup = (token, kycStatus) => {
    localStorage.setItem("token", token);
    localStorage.setItem("kyc_status", kycStatus || "pending");
    setView(kycStatus === "approved" ? "dashboard" : "kyc");
  };

  // Called by LoginPage after a successful /login response.
  // Login only succeeds when kyc_status is "approved" (backend enforces this),
  // but we still read the value from the server to be safe.
  const handleLogin = (kycStatus) => {
    localStorage.setItem("kyc_status", kycStatus || "approved");
    setView("dashboard");
  };

  // Called by KYCPage once the /kyc/submit call succeeds.
  const handleKYCComplete = () => {
    localStorage.setItem("kyc_status", "approved");
    setView("dashboard");
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("kyc_status");
    localStorage.removeItem("hasSeenGuide");
    setDashboardScreen("universe");
    setPortfolioStocks([]);
    setView("login");
  };

  const handleBuildPortfolio = (stocks) => {
    setPortfolioStocks(stocks);
    setDashboardScreen("optimizer");
  };

  if (view === "login") {
    return <LoginPage onLogin={handleLogin} onSignup={handleSignup} />;
  }

  if (view === "kyc") {
    return <KYCPage onKYCComplete={handleKYCComplete} />;
  }

  if (dashboardScreen === "optimizer") {
    return (
      <Dashboard
        onLogout={handleLogout}
        initialStocks={portfolioStocks}
        onBackToUniverse={() => setDashboardScreen("universe")}
      />
    );
  }

  return <UniverseDashboard onLogout={handleLogout} onBuildPortfolio={handleBuildPortfolio} />;
}

export default App;

