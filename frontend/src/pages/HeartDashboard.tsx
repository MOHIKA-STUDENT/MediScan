import { useState, useEffect } from 'react';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { 
  Heart, 
  Activity, 
  TrendingUp, 
  AlertTriangle, 
  Calendar, 
  Clock,
  User,
  Settings,
  Download,
  RefreshCw,
  Play,
  Pause,
  Brain,
  LogOut,
  Volume2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useNavigate } from 'react-router-dom';



const HeartbeatDashboard = () => {
  const [currentBPM, setCurrentBPM] = useState(72);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [heartbeatData, setHeartbeatData] = useState([]);
  const [alertLevel, setAlertLevel] = useState('normal'); // normal, warning, critical
const navigate = useNavigate();
  // Simulate real-time heartbeat data
  useEffect(() => {
    let interval;
    if (isMonitoring) {
      interval = setInterval(() => {
        const newBPM = 60 + Math.random() * 40 + Math.sin(Date.now() / 10000) * 15;
        const roundedBPM = Math.round(newBPM);
        setCurrentBPM(roundedBPM);
        
        // Update alert level based on BPM
        if (roundedBPM < 60 || roundedBPM > 100) {
          setAlertLevel('warning');
        } else if (roundedBPM < 50 || roundedBPM > 120) {
          setAlertLevel('critical');
        } else {
          setAlertLevel('normal');
        }

        // Add to heartbeat data for graph
        setHeartbeatData(prev => {
          const newData = [...prev, { time: Date.now(), bpm: roundedBPM }];
          return newData.slice(-50); // Keep last 50 readings
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isMonitoring]);

  const getAlertColor = () => {
    switch (alertLevel) {
      case 'warning': return 'text-yellow-500';
      case 'critical': return 'text-red-500';
      default: return 'text-green-500';
    }
  };

  const getAlertBadge = () => {
    switch (alertLevel) {
      case 'warning': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'critical': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      default: return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    }
  };

  const recentReadings = [
    { time: '14:30', bpm: 75, status: 'normal' },
    { time: '14:25', bpm: 78, status: 'normal' },
    { time: '14:20', bpm: 82, status: 'warning' },
    { time: '14:15', bpm: 71, status: 'normal' },
    { time: '14:10', bpm: 69, status: 'normal' }
  ];

  const healthMetrics = [
    { label: 'Avg BPM Today', value: '74', trend: '+2%', icon: Activity },
    { label: 'Max BPM', value: '95', trend: 'Normal', icon: TrendingUp },
    { label: 'Min BPM', value: '58', trend: 'Stable', icon: Heart },
    { label: 'Monitoring Time', value: '3h 42m', trend: 'Active', icon: Clock }
  ];

  const FloatingHeartbeat = () => (
    <div className="absolute inset-0 pointer-events-none">
      <Heart className={`absolute top-20 left-10 h-8 w-8 opacity-10 ${getAlertColor()} medical-pulse`} />
      <Activity className="absolute top-32 right-16 h-6 w-6 opacity-10 text-primary medical-float" style={{ animationDelay: '1s' }} />
      <Heart className={`absolute bottom-32 left-20 h-10 w-10 opacity-10 ${getAlertColor()} medical-pulse`} style={{ animationDelay: '2s' }} />
      <Activity className="absolute bottom-20 right-12 h-8 w-8 opacity-10 text-accent medical-float" style={{ animationDelay: '3s' }} />
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
      <FloatingHeartbeat />
      
      {/* Gradient Orbs */}
      <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-primary/10 rounded-full blur-3xl medical-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl medical-pulse" style={{ animationDelay: '1s' }} />

      {/* Header */}
      <header className="sticky top-0 z-40 glass-card border-b">
        <div className="relative z-10 w-full px-4 py-2">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
               <Button variant="outline" onClick={() => navigate(-1)}     className="mr-2">
    <span className="text-xl font-bold"> ← </span>
    </Button>
              <div className="p-2 bg-gradient-to-r from-primary to-accent rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold">HeartBeatMionitoring Dashboard</h1>
                <p className="text-sm text-muted-foreground">AI Heart Beat Monitoring </p>
              </div>
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline">
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onClick={() => {
                    localStorage.removeItem("token");
                    window.location.href = "/";
                  }}
                  className="cursor-pointer"
                >
                  <LogOut className="h-4 w-4 mr-2" />
                  Logout
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            
          </div>
          
        </div>
      </header>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Main BPM Display */}
        <div className="mb-8">
          <Card className="glass-card hover:shadow-glow transition-all duration-300">
            <CardContent className="p-8 text-center">
              <div className="flex items-center justify-center mb-6">
                <div className={`p-4 rounded-full bg-gradient-to-r from-red-500 to-pink-500 medical-pulse`}>
                  <Heart className="h-12 w-12 text-white" />
                </div>
              </div>
              <div className="space-y-4">
                <div>
                  <div className={`text-6xl font-bold ${getAlertColor()} mb-2`}>
                    {currentBPM}
                  </div>
                  <p className="text-xl text-muted-foreground">Beats per minute</p>
                </div>
                <Badge className={getAlertBadge()}>
                  {alertLevel.toUpperCase()} RANGE
                </Badge>
                <div className="flex justify-center space-x-4 pt-4">
                  <Button
                    onClick={() => setIsMonitoring(!isMonitoring)}
                    className={isMonitoring 
                      ? "bg-red-500 hover:bg-red-600 text-white" 
                      : "bg-green-500 hover:bg-green-600 text-white"
                    }
                  >
                    {isMonitoring ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                    {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
                  </Button>
                  <Button variant="outline">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Calibrate
                  </Button>
                  <Button variant="outline">
                    <Volume2 className="h-4 w-4 mr-2" />
                    Alerts
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Health Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {healthMetrics.map((metric, index) => (
            <Card key={index} className="glass-card hover:shadow-glow transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="p-2 bg-gradient-to-r from-primary to-accent rounded-lg">
                    <metric.icon className="h-5 w-5 text-white" />
                  </div>
                  <span className="text-xs text-green-600 font-medium">{metric.trend}</span>
                </div>
                <div>
                  <p className="text-2xl font-bold mb-1">{metric.value}</p>
                  <p className="text-sm text-muted-foreground">{metric.label}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Heartbeat Graph */}
          <Card className="glass-card hover:shadow-glow transition-all duration-300">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Activity className="h-5 w-5 mr-2 text-primary" />
                Live Heartbeat Waveform
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-48 bg-gradient-to-r from-background/50 to-accent/10 rounded-lg flex items-center justify-center">
                <div className="w-full h-32 relative overflow-hidden">
                  {/* Simulated ECG Wave */}
                  <svg className="w-full h-full" viewBox="0 0 400 100">
                    <defs>
                      <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8" />
                        <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.8" />
                      </linearGradient>
                    </defs>
                    <polyline
                      points="0,50 50,50 60,30 70,80 80,20 90,50 150,50 160,30 170,80 180,20 190,50 250,50 260,30 270,80 280,20 290,50 350,50 360,30 370,80 380,20 390,50 400,50"
                      fill="none"
                      stroke="url(#waveGradient)"
                      strokeWidth="2"
                      className={isMonitoring ? "animate-pulse" : ""}
                    />
                  </svg>
                </div>
              </div>
              <div className="flex justify-between items-center mt-4 text-sm text-muted-foreground">
                <span>0s</span>
                <span>Real-time ECG simulation</span>
                <span>30s</span>
              </div>
            </CardContent>
          </Card>

          {/* Recent Readings */}
          <Card className="glass-card hover:shadow-glow transition-all duration-300">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Clock className="h-5 w-5 mr-2 text-accent" />
                Recent Readings
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentReadings.map((reading, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-background/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-primary rounded-full medical-pulse" />
                      <span className="font-medium">{reading.time}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xl font-bold">{reading.bpm}</span>
                      <span className="text-sm text-muted-foreground">BPM</span>
                      <Badge 
                        variant="secondary" 
                        className={reading.status === 'normal' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}
                      >
                        {reading.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Alert Panel */}
        {alertLevel !== 'normal' && (
          <Card className="glass-card border-yellow-500/50 mt-8 animate-fade-in-up">
            <CardContent className="p-6">
              <div className="flex items-center space-x-4">
                <AlertTriangle className="h-8 w-8 text-yellow-500" />
                <div>
                  <h3 className="text-lg font-semibold">Heart Rate Alert</h3>
                  <p className="text-muted-foreground">
                    Your heart rate is {alertLevel === 'warning' ? 'outside normal range' : 'critically abnormal'}. 
                    Consider consulting a healthcare professional if this persists.
                  </p>
                </div>
                <Button variant="outline">
                  Dismiss
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Custom Styles */}
      <style>{`
        .medical-pulse {
          animation: medicalPulse 2s ease-in-out infinite;
        }
        
        .medical-float {
          animation: medicalFloat 6s ease-in-out infinite;
        }
        
        @keyframes medicalPulse {
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.05); }
        }
        
        @keyframes medicalFloat {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          33% { transform: translateY(-10px) rotate(5deg); }
          66% { transform: translateY(5px) rotate(-3deg); }
        }
        
        .animate-fade-in-up {
          animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default HeartbeatDashboard;