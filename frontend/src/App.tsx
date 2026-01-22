import { Toaster as Sonner } from "@/components/ui/sonner"; // using sonner toast only
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import MedVizioLanding from "./pages/MedVizioLanding";
import Login from "./pages/Login";
import SignUp from "./pages/SignUp";
import Dashboard from "./pages/Dashboard";
import ProtectedRoute from "./components/ProtectedRoute";
import EditProfile from "./pages/EditProfile";
import HeartbeatDashboard from "./pages/HeartDashboard";


const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Sonner position="top-right" richColors /> {/* ✅ Fix here */}
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MedVizioLanding />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/edit-profile" element={
            <ProtectedRoute>
              <EditProfile />
            </ProtectedRoute>} />
          <Route path="/dashboard" element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } />
          <Route path="/heartbeat" element={
  <ProtectedRoute>
    <HeartbeatDashboard />
  </ProtectedRoute>
} />

        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
