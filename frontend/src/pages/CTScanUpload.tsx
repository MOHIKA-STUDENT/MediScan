import { useState } from "react"; 
import { Upload, Plus, Brain, AlertCircle, CheckCircle, BarChart3, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface AnalysisResult {
  filename: string;
  predicted_class: string;
  confidence: number;
  all_probabilities: Record<string, number>;
  model_agreement?: number;
  ensemble_size?: number;
  error?: string;
  ai_analysis?: string;
  ai_provider?: string;
  has_ai_report?: boolean;
}

const CTScanUpload = () => {
  const [selectedCTFiles, setSelectedCTFiles] = useState<File[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);

  const BACKEND_URL = 'http://127.0.0.1:5000';

  const handleAnalyze = async () => {
    if (selectedCTFiles.length === 0) {
      toast.error('Please upload at least one CT scan image');
      return;
    }

    setIsAnalyzing(true);
    const toastId = toast.loading("Analyzing CT scan images with AI ensemble...");

    try {
      const formData = new FormData();
      
      selectedCTFiles.forEach((file) => {
        formData.append('files', file);
      });

      const response = await fetch(`${BACKEND_URL}/ensemble_analyze_with_ai`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      setAnalysisResults(result.results);

      toast.success(`Analysis complete! Processed ${result.total_files || result.results.length} files`, {
        id: toastId
      });

    } catch (error) {
      console.error('Analysis error:', error);
      toast.error('Analysis failed. Please check if the server is running.', {
        id: toastId
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const resetAnalysis = () => {
    setAnalysisResults([]);
    setSelectedCTFiles([]);
  };

  // Format AI analysis for better display
  const formatAnalysis = (text: string) => {
    if (!text) return [];
    
    // Split by double newlines to get sections
    const sections = text.split('\n\n').filter(s => s.trim());
    
    return sections.map((section, idx) => {
      const lines = section.split('\n').filter(l => l.trim());
      
      // Check if first line is a section header (all caps or title-like)
      const firstLine = lines[0];
      const isHeader = firstLine === firstLine.toUpperCase() || 
                       (!firstLine.startsWith('•') && !firstLine.includes(':') && lines.length > 1);
      
      if (isHeader) {
        return {
          type: 'section',
          header: firstLine,
          content: lines.slice(1),
          key: idx
        };
      } else {
        return {
          type: 'paragraph',
          content: lines,
          key: idx
        };
      }
    });
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-2">AI-Powered CT Scan Analysis</h2>
        <p className="text-muted-foreground">
          Upload your CT scan images for intelligent kidney condition analysis using advanced AI ensemble
        </p>
      </div>

      {/* Upload Area */}
      <div className="relative">
        <input
          type="file"
          multiple
          accept=".jpeg,.jpg,.png"
          onChange={(e) => {
            const files = e.target.files;
            if (files && files.length > 0) {
              setSelectedCTFiles((prev) => [...prev, ...Array.from(files)]);
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
              <h3 className="text-xl font-semibold">Drag & drop your CT scan images here</h3>
              <p className="text-muted-foreground">Or click to browse files</p>
              <Button
                type="button"
                className="bg-gradient-to-r from-primary to-accent text-white pointer-events-none"
              >
                <Plus className="mr-2 h-4 w-4" /> Choose Files
              </Button>
              <div className="text-sm text-muted-foreground">
                Supports: JPEG, PNG • Max size: 50MB per file
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Uploaded Files */}
      {selectedCTFiles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Uploaded Files ({selectedCTFiles.length})</span>
              <Button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="bg-gradient-to-r from-primary to-accent text-white"
              >
                {isAnalyzing ? (
                  <>
                    <Brain className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="mr-2 h-4 w-4" />
                    Analyze with AI
                  </>
                )}
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4">
              {selectedCTFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 rounded-lg border bg-background"
                >
                  <div className="flex items-center space-x-4">
                    <img
                      src={URL.createObjectURL(file)}
                      alt={file.name}
                      className="h-16 w-16 object-cover rounded-md border"
                    />
                    <div>
                      <p className="font-medium text-sm">{file.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {(file.size / 1024 / 1024).toFixed(1)} MB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() =>
                      setSelectedCTFiles((prev) => prev.filter((_, i) => i !== index))
                    }
                    className="text-red-500 hover:text-red-700 text-xl font-bold h-8 w-8 flex items-center justify-center rounded-full hover:bg-red-50"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Analysis Results */}
      {analysisResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              AI Ensemble Analysis Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6">
              {analysisResults.map((result, index) => (
                <div key={index} className="p-4 border rounded-lg">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h4 className="font-semibold">{result.filename}</h4>
                      {result.error ? (
                        <div className="flex items-center gap-2 text-red-600 mt-2">
                          <AlertCircle className="h-4 w-4" />
                          <span className="text-sm">{result.error}</span>
                        </div>
                      ) : (
                        <div className="mt-2">
                          <div className="flex items-center gap-3 mb-2">
                            <Badge className={getConfidenceBadge(result.confidence)}>
                              {result.predicted_class}
                            </Badge>
                            <span className={`font-semibold ${getConfidenceColor(result.confidence)}`}>
                              {(result.confidence * 100).toFixed(1)}% confidence
                            </span>
                          </div>
                          <Progress value={result.confidence * 100} className="h-2 mb-2" />
                          
                          {result.model_agreement !== undefined && (
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <span>Model Agreement: {(result.model_agreement * 100).toFixed(0)}%</span>
                              {result.ensemble_size && (
                                <span>• {result.ensemble_size} models used</span>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {!result.error && (
                    <div className="mt-4">
                      <h5 className="font-medium mb-2 flex items-center gap-2">
                        <BarChart3 className="h-4 w-4" />
                        All Class Probabilities
                      </h5>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {Object.entries(result.all_probabilities).map(([className, probability]) => (
                          <div key={className} className="text-center p-2 bg-gray-50 rounded">
                            <div className="text-sm font-medium">{className}</div>
                            <div className="text-xs text-muted-foreground">
                              {(probability * 100).toFixed(1)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* AI Report Section - Properly Formatted */}
                  {!result.error && result.has_ai_report && result.ai_analysis && (
                    <Collapsible className="mt-6">
                      <CollapsibleTrigger asChild>
                        <Button variant="outline" className="w-full justify-between">
                          <div className="flex items-center">
                            <Brain className="mr-2 h-4 w-4" />
                            <span>View AI Medical Report ({result.ai_provider})</span>
                          </div>
                          <ChevronDown className="h-4 w-4 transition-transform duration-200" />
                        </Button>
                      </CollapsibleTrigger>
                      <CollapsibleContent className="mt-4">
                        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-6 space-y-6">
                          {formatAnalysis(result.ai_analysis).map((item) => (
                            <div key={item.key}>
                              {item.type === 'section' ? (
                                <div>
                                  <h3 className="text-lg font-bold text-gray-900 mb-3 pb-2 border-b-2 border-blue-300">
                                    {item.header}
                                  </h3>
                                  <div className="space-y-2">
                                    {item.content.map((line, idx) => (
                                      <p key={idx} className="text-sm text-gray-700 leading-relaxed pl-2">
                                        {line}
                                      </p>
                                    ))}
                                  </div>
                                </div>
                              ) : (
                                <div className="space-y-2">
                                  {item.content.map((line, idx) => (
                                    <p key={idx} className="text-sm text-gray-700 leading-relaxed pl-2">
                                      {line}
                                    </p>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                          
                          {result.ai_provider?.includes('Template') && (
                            <div className="mt-4 pt-4 border-t border-blue-200">
                              <p className="text-xs text-gray-500 italic">
                                ℹ️ Enhanced template used (AI APIs temporarily unavailable). For AI-powered analysis, configure API keys in .env file.
                              </p>
                            </div>
                          )}
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  )}
                </div>
              ))}
            </div>
            <div className="mt-6 flex gap-2">
              <Button onClick={resetAnalysis} variant="outline">
                <Upload className="mr-2 h-4 w-4" />
                Analyze More Images
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default CTScanUpload;