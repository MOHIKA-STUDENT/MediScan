import { useState } from 'react';
import { Upload, FileText, History, User, Settings, Brain, Plus, Search, Filter } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { LogOut } from "lucide-react";
import Profile from "./Profile"; // adjust path as needed


const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const sidebarItems = [
    { id: 'upload', icon: Upload, label: 'Upload Report', badge: null },
    { id: 'reports', icon: FileText, label: 'My Reports', badge: '12' },
    { id: 'history', icon: History, label: 'History', badge: null },
    { id: 'profile', icon: User, label: 'Profile', badge: null },
  ];

  const recentReports = [
    {
      id: 1,
      name: 'Blood Test - Complete Panel',
      date: '2024-01-15',
      status: 'analyzed',
      type: 'Blood Test',
      riskLevel: 'low'
    },
    {
      id: 2,
      name: 'CT Scan - Chest',
      date: '2024-01-10',
      status: 'analyzing',
      type: 'CT Scan',
      riskLevel: 'medium'
    },
    {
      id: 3,
      name: 'MRI - Brain',
      date: '2024-01-05',
      status: 'analyzed',
      type: 'MRI',
      riskLevel: 'low'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'analyzed': return 'bg-accent text-accent-foreground';
      case 'analyzing': return 'bg-primary text-primary-foreground';
      case 'pending': return 'bg-muted text-muted-foreground';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'high': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const renderUploadSection = () => (
  <div className="space-y-8">
    <div>
      <h2 className="text-3xl font-bold font-heading mb-2">Upload Medical Report</h2>
      <p className="text-muted-foreground">Upload your medical reports for instant AI analysis</p>
    </div>

    {/* Upload Area */}
    <div className="relative">
      <input
        type="file"
        id="file-upload"
        multiple
        accept=".pdf,.jpeg,.jpg,.png,.dcm"
        onChange={(e) => {
          const files = e.target.files;
          if (files && files.length > 0) {
            setSelectedFiles((prev) => [...prev, ...Array.from(files)]);
          }
        }}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
      />
      <Card className="border-dashed border-2 border-primary/30 hover:border-primary/50 transition-colors relative z-0">
        <CardContent className="p-12 text-center pointer-events-none">
          <div className="flex flex-col items-center space-y-6">
            <div className="p-6 bg-gradient-to-r from-primary to-accent rounded-full">
              <Upload className="h-12 w-12 text-white" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-semibold">Drag & drop your files here</h3>
              <p className="text-muted-foreground">Or click to browse files</p>
            </div>
            <Button
              type="button"
              className="bg-gradient-to-r from-primary to-accent text-white pointer-events-none"
            >
              <Plus className="mr-2 h-4 w-4" />
              Choose Files
            </Button>
            <div className="text-sm text-muted-foreground">
              Supports: PDF, JPEG, PNG • Max size: 50MB
            </div>

            
          </div>

          
        </CardContent>
        
      </Card>
      
    </div>

    {/* Uploaded Files Section */}
    {selectedFiles.length > 0 && (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Uploaded Files</h3>
        <ul className="space-y-2">
          {selectedFiles.map((file, index) => (
            <li
              key={index}
              className="flex items-center justify-between p-3 rounded-md border bg-background"
            >
              <div className="flex items-center space-x-4">
                {file.type.startsWith("image/") ? (
                  <img
                    src={URL.createObjectURL(file)}
                    alt={file.name}
                    className="h-16 w-16 object-cover rounded-md"
                  />
                ) : (
                  <FileText className="h-8 w-8 text-muted-foreground" />
                )}
                <div>
                  <p className="font-medium text-sm">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>

              <button
                onClick={(e) => {
                  e.stopPropagation(); // Prevent file picker from opening
                  setSelectedFiles((prev) =>
                    prev.filter((_, i) => i !== index)
                  );
                }}
                className="text-red-500 hover:text-red-700 text-xl font-bold"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </div>
    )}
  </div>
);




  const renderReportsSection = () => (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold font-heading mb-2">My Reports</h2>
          <p className="text-muted-foreground">View and manage your medical reports</p>
        </div>
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search reports..." className="pl-10 w-64" />
          </div>
          <Button variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Filter
          </Button>
        </div>
      </div>

      <div className="grid gap-4">
        {recentReports.map((report) => (
          <Card key={report.id} className="hover:shadow-glow transition-all duration-300 cursor-pointer">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3 className="text-lg font-semibold">{report.name}</h3>
                    <Badge className={getStatusColor(report.status)}>
                      {report.status}
                    </Badge>
                    <Badge variant="outline" className={getRiskColor(report.riskLevel)}>
                      {report.riskLevel} risk
                    </Badge>
                  </div>
                  <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                    <span>Type: {report.type}</span>
                    <span>Date: {report.date}</span>
                  </div>
                </div>
                <Button variant="outline">
                  View Report
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-card border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-gradient-to-r from-primary to-accent rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold font-heading">MediScan Dashboard</h1>
                <p className="text-sm text-muted-foreground">AI Medical Report Analysis</p>
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
        window.location.href = "/"; // Or use `navigate("/login")` if you're using React Router
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

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar */}
          <div className="lg:w-64 space-y-2">
            {sidebarItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
               className={`w-full flex items-center justify-between px-4 py-3 rounded-lg text-left transition-all duration-200 transform
  ${activeTab === item.id
    ? 'bg-gradient-to-r from-primary to-accent text-white shadow-glow hover:brightness-105 hover:scale-[1.02]'
    : 'bg-background text-foreground hover:bg-muted hover:scale-[1.02]'}
`}

              >
                <div className="flex items-center space-x-3">
                  <item.icon className="h-5 w-5" />
                  <span className="font-medium">{item.label}</span>
                </div>
                {item.badge && (
                  <Badge variant="secondary" className="ml-2">
                    {item.badge}
                  </Badge>
                )}
              </button>
            ))}
          </div>

          {/* Main Content */}
          <div className="flex-1">
            {activeTab === 'upload' && renderUploadSection()}
            {activeTab === 'reports' && renderReportsSection()}
            {activeTab === 'history' && (
              <div className="text-center py-12">
                <h2 className="text-2xl font-bold font-heading mb-4">History</h2>
                <p className="text-muted-foreground">View your analysis history</p>
              </div>
            )}
            {activeTab === 'profile' && <Profile />}

          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;