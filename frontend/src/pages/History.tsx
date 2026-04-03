import { useState, useEffect } from "react";
import { format } from "date-fns";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Activity, Clock, FileText, Image as ImageIcon, ChevronRight, CheckCircle, TrendingUp, TrendingDown, Trash2 } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { toast } from "sonner";

interface Report {
  _id: string;
  reportType: string;
  fileUrl: string;
  analysisResult: any;
  createdAt: string;
}

const History = () => {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const token = localStorage.getItem("token");
      if (!token) return;

      const res = await fetch("http://localhost:5001/api/reports/history", {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      const data = await res.json();
      setReports(data);
    } catch (err) {
      console.error("Failed to fetch reports:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation(); // prevent modal opening
    const confirmDelete = window.confirm("Are you sure you want to delete this report history?");
    if (!confirmDelete) return;

    try {
      const token = localStorage.getItem("token");
      const res = await fetch(`http://localhost:5001/api/reports/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });

      if (res.ok) {
        setReports(reports.filter(r => r._id !== id));
        toast.success("Report deleted successfully");
        if (selectedReport?._id === id) setSelectedReport(null);
      } else {
        toast.error("Failed to delete report");
      }
    } catch (err) {
      console.error(err);
      toast.error("Error deleting report");
    }
  };

  const getStatusBadge = (status: string) => {
    const isPositive = status?.toLowerCase() === 'high' || status?.toLowerCase() === 'low';
    return (
      <Badge variant={isPositive ? "destructive" : "default"} className={isPositive ? "bg-red-500" : "bg-green-500"}>
        {status}
      </Badge>
    );
  };

  const renderReportModal = () => {
    if (!selectedReport) return null;

    const isBloodReport = selectedReport.reportType === "Blood Report";

    return (
      <Dialog open={!!selectedReport} onOpenChange={() => setSelectedReport(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl">
              {isBloodReport ? <Activity className="text-red-500" /> : <ImageIcon className="text-blue-500" />}
              {selectedReport.reportType} Details
            </DialogTitle>
            <DialogDescription>
              Uploaded on {format(new Date(selectedReport.createdAt), "PPP")}
            </DialogDescription>
          </DialogHeader>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
            <div className="bg-muted/30 rounded-lg border p-2 flex flex-col justify-center items-center h-[500px] overflow-hidden relative group">
              <img 
                 src={selectedReport.fileUrl.replace(/\.pdf/i, '.png')} 
                 alt="Report preview" 
                 className="max-w-full max-h-full object-contain rounded" 
              />
              
              {selectedReport.fileUrl.toLowerCase().includes(".pdf") && (
                <div className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                  <a 
                    href={selectedReport.fileUrl.replace('http:', 'https:')}
                    download
                    className="bg-primary text-primary-foreground px-4 py-2 rounded-lg inline-flex items-center gap-2 shadow-lg hover:bg-primary/90"
                  >
                    <FileText className="w-4 h-4" /> Download PDF
                  </a>
                </div>
              )}
            </div>

            {/* Analysis Results Summary */}
            <div className="space-y-6 overflow-y-auto pr-2 max-h-[500px]">
              {isBloodReport ? (
                <div>
                  <h3 className="text-lg font-bold border-b pb-2 mb-4">Detected Conditions</h3>
                  {selectedReport.analysisResult?.positive_diseases?.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                       {selectedReport.analysisResult.positive_diseases.map((d: string) => (
                         <Badge key={d} variant="destructive">{d}</Badge>
                       ))}
                    </div>
                  ) : (
                    <Badge className="bg-green-500">Normal (No Critical Conditions Detected)</Badge>
                  )}
                  
                  {selectedReport.analysisResult?.ai_analysis && (
                    <div className="mt-6 bg-blue-50 p-4 rounded-lg text-sm text-gray-700 whitespace-pre-wrap">
                      <p className="font-semibold text-blue-900 mb-2">Full AI Analysis:</p>
                      {selectedReport.analysisResult.ai_analysis}
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <h3 className="text-lg font-bold border-b pb-2 mb-4">CT Scan Prediction</h3>
                  <div className="flex items-center gap-4 p-4 bg-muted/30 rounded-lg mb-6">
                    <div className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-indigo-600">
                      {selectedReport.analysisResult.predicted_class}
                    </div>
                    <Badge variant="outline" className="text-lg">
                       {(selectedReport.analysisResult.confidence * 100).toFixed(1)}% Match
                    </Badge>
                  </div>
                  
                  {selectedReport.analysisResult?.ai_analysis && (
                     <div className="bg-blue-50 p-4 rounded-lg text-sm text-gray-700 whitespace-pre-wrap">
                       <p className="font-semibold text-blue-900 mb-2">AI Insights:</p>
                       {selectedReport.analysisResult.ai_analysis}
                     </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    );
  };

  return (
    <div className="space-y-8 animate-in fade-in zoom-in duration-500">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Report History</h2>
        <p className="text-muted-foreground mt-2">
          View all your previously uploaded medical reports and their AI analysis results.
        </p>
      </div>

      {loading ? (
        <div className="flex justify-center p-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      ) : reports.length === 0 ? (
        <Card className="border-dashed border-2 bg-muted/10">
          <CardContent className="flex flex-col items-center justify-center p-16 text-center">
            <Clock className="w-12 h-12 text-muted-foreground mb-4 opacity-50" />
            <h3 className="text-xl font-semibold">No history found</h3>
            <p className="text-muted-foreground mt-2 max-w-sm">
              You haven't uploaded any reports yet. Upload a Blood Report or CT Scan to see your history here.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {reports.map((report) => {
            const isBlood = report.reportType === "Blood Report";
            const date = new Date(report.createdAt);
            
            // Generate quick summary string
            let summaryStr = "Processed successfully";
            if (isBlood && report.analysisResult?.positive_diseases?.length > 0) {
              summaryStr = `Detected: ${report.analysisResult.positive_diseases.join(', ')}`;
            } else if (!isBlood && report.analysisResult?.predicted_class) {
              summaryStr = `Result: ${report.analysisResult.predicted_class} (${(report.analysisResult.confidence * 100).toFixed(0)}%)`;
            }

            return (
              <Card key={report._id} className="hover:shadow-md transition-shadow group overflow-hidden border-t-4 hover:-translate-y-1 duration-200" style={{ borderTopColor: isBlood ? '#ef4444' : '#3b82f6' }}>
                <CardHeader className="pb-3">
                  <div className="flex justify-between items-start">
                    <div className="p-2 rounded-lg bg-muted text-foreground mb-2">
                       {isBlood ? <Activity className="w-5 h-5 text-red-500" /> : <ImageIcon className="w-5 h-5 text-blue-500" />}
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs font-mono">
                        {format(date, "MMM dd, yyyy")}
                      </Badge>
                      <button 
                        onClick={(e) => handleDelete(e, report._id)}
                        className="text-muted-foreground hover:text-red-500 transition-colors p-1"
                        title="Delete Report"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  <CardTitle className="text-lg">{report.reportType}</CardTitle>
                  <CardDescription className="line-clamp-1">{summaryStr}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    variant="ghost" 
                    className="w-full justify-between mt-2 group-hover:bg-primary/5"
                    onClick={() => setSelectedReport(report)}
                  >
                    View Full Analysis
                    <ChevronRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {renderReportModal()}
    </div>
  );
};

export default History;
