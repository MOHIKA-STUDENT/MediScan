import { useState } from "react";
import { Upload, FileText, Plus, Brain, Image as ImageIcon, FileUp, Activity, AlertCircle, CheckCircle, TrendingUp, TrendingDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

interface BloodData {
  hemoglobin: string;
  wbc: string;
  rbc: string;
  platelets: string;
  glucose: string;
  creatinine: string;
  urea: string;
  sodium: string;
  potassium: string;
  calcium: string;
}

interface ExtractedValue {
  name: string;
  value: string;
  unit: string;
  status: 'normal' | 'high' | 'low';
  range?: string;
}

const MedicalReportUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [bloodData, setBloodData] = useState<BloodData>({
    hemoglobin: '', wbc: '', rbc: '', platelets: '', glucose: '',
    creatinine: '', urea: '', sodium: '', potassium: '', calcium: ''
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiReport, setAiReport] = useState('');
  const [aiProvider, setAiProvider] = useState('');
  const [analysisMode, setAnalysisMode] = useState<'file' | 'manual'>('file');
  const [error, setError] = useState('');
  const [extractedValues, setExtractedValues] = useState<ExtractedValue[]>([]);
  const [showValues, setShowValues] = useState(true);

  // Add this new state at the top with other useState declarations (around line 27)
const [analysisResult, setAnalysisResult] = useState<any>(null);

  const BACKEND_URL = 'http://127.0.0.1:5000';

  // Parse extracted values from AI report

  // 🔧 FIXED: Parse extracted values from AI report with correct status detection
  const parseExtractedValues = (text: string): ExtractedValue[] => {
    const values: ExtractedValue[] = [];
    const lines = text.split('\n');
    let inExtractedSection = false;

    for (const line of lines) {
      if (line.includes('EXTRACTED VALUES')) {
        inExtractedSection = true;
        continue;
      }
      if (inExtractedSection && line.match(/^[A-Z]/)) {
        if (!line.includes(':')) {
          inExtractedSection = false;
          break;
        }
      }
      
      if (inExtractedSection && line.includes(':')) {
        const match = line.match(/(.+?):\s*([0-9.]+)\s*(.+)?/);
        if (match) {
          const name = match[1].replace(/^•\s*/, '').trim();
          const value = match[2].trim();
          const restOfLine = match[3]?.trim() || '';
          
          let status: 'normal' | 'high' | 'low' = 'normal';
          let unit = '';
          let range = '';
          
          // 🎯 CRITICAL FIX: Extract status from backend "current status: X"
          const statusMatch = restOfLine.match(/current status:\s*(low|high|normal)/i);
          if (statusMatch) {
            status = statusMatch[1].toLowerCase() as 'normal' | 'high' | 'low';
          }
          
          // Extract normal range
          const rangeMatch = restOfLine.match(/normal range:\s*([^,]+)/i);
          if (rangeMatch) {
            range = rangeMatch[1].trim();
          }
          
          // Extract unit
          const unitMatch = restOfLine.match(/^([^(]+?)(?:\(|$)/);
          if (unitMatch) {
            unit = unitMatch[1].trim();
          }
          
          // Fallback: If no status from backend, determine from value
          if (!statusMatch) {
            const numValue = parseFloat(value);
            
            if (name.toLowerCase().includes('wbc') || name.toLowerCase().includes('leucocyte')) {
              if (numValue < 4000) status = 'low';
              else if (numValue > 11000) status = 'high';
              if (!range) range = '4-11 ×10³/µL';
            } else if (name.toLowerCase().includes('platelet')) {
              if (numValue < 150000) status = 'low';
              else if (numValue > 400000) status = 'high';
              if (!range) range = '150-400 ×10³/µL';
            } else if (name.toLowerCase().includes('hemoglobin')) {
              if (numValue < 13) status = 'low';
              else if (numValue > 17) status = 'high';
              if (!range) range = '13-17 g/dL';
            } else if (name.toLowerCase().includes('rdw')) {
              if (numValue > 14.5) status = 'high';
              if (!range) range = '11.5-14.5%';
            } else if (name.toLowerCase().includes('neutrophil')) {
              if (numValue < 45) status = 'low';
              else if (numValue > 70) status = 'high';
              if (!range) range = '45-70%';
            } else if (name.toLowerCase().includes('lymphocyte')) {
              if (numValue < 20) status = 'low';
              else if (numValue > 40) status = 'high';
              if (!range) range = '20-40%';
            }
          }

          values.push({ name, value, unit, status, range });
        }
      }
    }

    return values;
  };
  const formatAnalysis = (text: string) => {
    if (!text) return [];
    const sections = text.split('\n\n').filter(s => s.trim());
    return sections.map((section, idx) => {
      const lines = section.split('\n').filter(l => l.trim());
      const firstLine = lines[0];
      const isHeader = firstLine === firstLine.toUpperCase() || 
                       (!firstLine.startsWith('•') && !firstLine.includes(':') && lines.length > 1);
      
      if (isHeader) {
        return { type: 'section', header: firstLine, content: lines.slice(1), key: idx };
      }
      return { type: 'paragraph', content: lines, key: idx };
    });
  };

  const getStatusBadge = (status: string) => {
    const configs = {
      normal: {
        bg: 'bg-green-100',
        text: 'text-green-800',
        icon: <CheckCircle className="w-3 h-3" />,
        label: 'Normal'
      },
      high: {
        bg: 'bg-red-100',
        text: 'text-red-800',
        icon: <TrendingUp className="w-3 h-3" />,
        label: 'High'
      },
      low: {
        bg: 'bg-orange-100',
        text: 'text-orange-800',
        icon: <TrendingDown className="w-3 h-3" />,
        label: 'Low'
      }
    };

    const config = configs[status] || configs.normal;

    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${config.bg} ${config.text}`}>
        {config.icon}
        {config.label}
      </span>
    );
  };

  const handleFileAnalyze = async () => {
  if (selectedFiles.length === 0) {
    setError('Please upload at least one file (PDF or Image)');
    return;
  }

  setIsAnalyzing(true);
  setError('');

  try {
    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));

    const response = await fetch(`${BACKEND_URL}/api/blood-report/analyze-pdf`, {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Analysis failed');
    }

    const analysis = result.ai_analysis || 'No analysis available';
    setAiReport(analysis);
    setAiProvider(result.ai_provider || 'Unknown');
    setExtractedValues(parseExtractedValues(analysis));
    setAnalysisResult(result);  // ⭐ ADD THIS LINE
    setError('');
  } catch (err: any) {
    console.error('File analysis error:', err);
    setError(err.message || 'Analysis failed. Check if backend is running and API keys are configured.');
    setAiReport('');
    setExtractedValues([]);
    setAnalysisResult(null);  // ⭐ ADD THIS LINE
  } finally {
    setIsAnalyzing(false);
  }
};

  const handleManualAnalyze = async () => {
  const validData = Object.fromEntries(
    Object.entries(bloodData).filter(([_, v]) => v.trim() !== '')
  );
  
  if (Object.keys(validData).length === 0) {
    setError('Enter at least one blood parameter');
    return;
  }

  setIsAnalyzing(true);
  setError('');

  try {
    const response = await fetch(`${BACKEND_URL}/api/blood-report/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(validData),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Analysis failed');
    }

    const analysis = result.ai_analysis || 'No analysis available';
    setAiReport(analysis);
    setAiProvider(result.ai_provider || 'Unknown');
    setExtractedValues(parseExtractedValues(analysis));
    setAnalysisResult(result);  // ⭐ ADD THIS LINE
    setError('');
  } catch (err: any) {
    console.error('Manual analysis error:', err);
    setError(err.message || 'Analysis failed. Check backend connection.');
    setAiReport('');
    setExtractedValues([]);
    setAnalysisResult(null);  // ⭐ ADD THIS LINE
  } finally {
    setIsAnalyzing(false);
  }
};

  const handleAnalyze = () => {
    analysisMode === 'file' ? handleFileAnalyze() : handleManualAnalyze();
  };

  const reset = () => {
  setSelectedFiles([]);
  setBloodData({
    hemoglobin: '', wbc: '', rbc: '', platelets: '', glucose: '',
    creatinine: '', urea: '', sodium: '', potassium: '', calcium: ''
  });
  setAiReport('');
  setAiProvider('');
  setExtractedValues([]);
  setAnalysisResult(null);  // ⭐ ADD THIS LINE
  setError('');
};

  const getFileIcon = (filename: string) => {
    const lower = filename.toLowerCase();
    if (lower.endsWith('.pdf')) return <FileText className="h-8 w-8 text-red-500" />;
    if (lower.match(/\.(jpg|jpeg|png|bmp|tiff)$/)) return <ImageIcon className="h-8 w-8 text-blue-500" />;
    return <FileUp className="h-8 w-8 text-gray-500" />;
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-2">Blood Report AI Analysis</h2>
        <p className="text-muted-foreground">
          Upload PDF reports or images, or enter values manually for AI-powered health insights.
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="flex gap-4">
        <Button
          variant={analysisMode === 'file' ? 'default' : 'outline'}
          onClick={() => setAnalysisMode('file')}
          className="flex-1"
        >
          <Upload className="mr-2 h-4 w-4" />
          File Upload (PDF/Image)
        </Button>
        <Button
          variant={analysisMode === 'manual' ? 'default' : 'outline'}
          onClick={() => setAnalysisMode('manual')}
          className="flex-1"
        >
          <FileText className="mr-2 h-4 w-4" />
          Manual Entry
        </Button>
      </div>

      {/* Error Display */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4">
            <p className="text-red-600 text-sm">⚠️ {error}</p>
          </CardContent>
        </Card>
      )}

      {analysisMode === 'file' ? (
        <>
          {/* File Upload */}
          <div className="relative">
            <input
              type="file"
              multiple
              accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff"
              onChange={(e) => {
                const files = Array.from(e.target.files || []);
                setSelectedFiles(prev => [...prev, ...files]);
                setError('');
              }}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
            />
            <Card className="border-dashed border-2 border-primary/30 hover:border-primary/50 transition-colors">
              <CardContent className="p-12 text-center pointer-events-none">
                <div className="flex flex-col items-center space-y-6">
                  <div className="p-6 bg-gradient-to-r from-primary to-accent rounded-full">
                    <Upload className="h-12 w-12 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold">Drag & drop blood reports here</h3>
                  <p className="text-muted-foreground">PDF documents or images (JPG, PNG, etc.)</p>
                  <Button className="bg-gradient-to-r from-primary to-accent text-white pointer-events-none">
                    <Plus className="mr-2 h-4 w-4" /> Choose Files
                  </Button>
                  <div className="text-sm text-muted-foreground">
                    Supports: PDF, JPG, PNG, BMP, TIFF • Max 50MB per file
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Uploaded Files */}
          {selectedFiles.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Uploaded Files ({selectedFiles.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {selectedFiles.map((file, idx) => (
                    <li key={idx} className="flex items-center justify-between p-3 rounded-md border bg-background">
                      <div className="flex items-center space-x-4">
                        {getFileIcon(file.name)}
                        <div>
                          <p className="font-medium text-sm">{file.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {(file.size / 1024 / 1024).toFixed(1)} MB
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => setSelectedFiles(prev => prev.filter((_, i) => i !== idx))}
                        className="text-red-500 hover:text-red-700 text-xl font-bold"
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </>
      ) : (
        <>
          {/* Manual Entry */}
          <Card>
            <CardHeader>
              <CardTitle>Enter Blood Test Values</CardTitle>
              <p className="text-sm text-muted-foreground">Provide numerical values</p>
            </CardHeader>
            <CardContent className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {Object.entries(bloodData).map(([key, value]) => (
                <div key={key} className="space-y-1">
                  <label className="text-xs font-medium capitalize">{key.replace('_', ' ')}</label>
                  <Input
                    placeholder={`e.g., ${key === 'hemoglobin' ? '12.5' : '7500'}`}
                    value={value}
                    onChange={(e) => setBloodData({ ...bloodData, [key]: e.target.value })}
                    type="number"
                    step="0.01"
                  />
                </div>
              ))}
            </CardContent>
          </Card>
        </>
      )}

      {/* Analyze Button */}
      <Button
        onClick={handleAnalyze}
        disabled={isAnalyzing}
        className="bg-gradient-to-r from-primary to-accent text-white w-full"
      >
        {isAnalyzing ? (
          <>
            <Brain className="mr-2 h-4 w-4 animate-spin" />
            Analyzing with AI...
          </>
        ) : (
          <>
            <Brain className="mr-2 h-4 w-4" />
            Analyze with AI
          </>
        )}
      </Button>

      {/* Results Section */}



      {aiReport && (
        <div className="space-y-6">
          {/* Disease Classification Results */}
{analysisResult?.disease_predictions && (
  <Card>
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Activity className="h-5 w-5" />
        ML Disease Classification
        <span className="text-sm font-normal text-muted-foreground">
          (6 RandomForest Models)
        </span>
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div className="space-y-3">
        {Object.entries(analysisResult.disease_predictions.predictions).map(([disease, pred]: [string, any]) => (
          <div
            key={disease}
            className={`flex items-center justify-between p-3 rounded-lg ${
              pred.is_positive 
                ? 'bg-red-50 border border-red-200' 
                : 'bg-green-50 border border-green-200'
            }`}
          >
            <div className="flex-1">
              <h3 className="font-semibold text-sm">{disease}</h3>
              <p className="text-xs text-gray-600">
                Confidence: {(pred.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-right">
              {pred.is_positive ? (
                <span className="px-3 py-1 bg-red-600 text-white rounded-full text-xs font-bold">
                  POSITIVE
                </span>
              ) : (
                <span className="px-3 py-1 bg-green-600 text-white rounded-full text-xs font-bold">
                  NEGATIVE
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
      
      {analysisResult.positive_diseases && analysisResult.positive_diseases.length > 0 && (
        <div className="mt-4 p-4 bg-red-100 border-l-4 border-red-500 rounded">
          <p className="text-sm font-semibold text-red-800">
            ⚠️ Detected Conditions: {analysisResult.positive_diseases.join(', ')}
          </p>
          <p className="text-xs text-red-700 mt-1">
            Please consult a healthcare professional for confirmation.
          </p>
        </div>
      )}
    </CardContent>
  </Card>
)}
          {/* Extracted Values with Status */}
          {extractedValues.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Key Blood Values
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowValues(!showValues)}
                  >
                    {showValues ? 'Hide' : 'Show'}
                  </Button>
                </div>
              </CardHeader>
              {showValues && (
                <CardContent>
                  <div className="space-y-3">
                    {extractedValues.map((item, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                      >
                        <div className="flex-1">
                          <h3 className="font-semibold text-sm text-gray-800">{item.name}</h3>
                          {item.range && (
                            <p className="text-xs text-gray-500">Normal: {item.range}</p>
                          )}
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="text-right">
                            <p className="text-base font-bold text-gray-800">
                              {item.value} <span className="text-xs font-normal text-gray-600">{item.unit}</span>
                            </p>
                          </div>
                          {getStatusBadge(item.status)}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Summary Stats */}
                  <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t">
                    <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
                      <div className="flex items-center justify-center gap-1 mb-1">
                        <CheckCircle className="w-4 h-4 text-green-600" />
                        <span className="text-xs font-semibold text-green-800">Normal</span>
                      </div>
                      <p className="text-xl font-bold text-green-600">
                        {extractedValues.filter(v => v.status === 'normal').length}
                      </p>
                    </div>
                    <div className="text-center p-3 bg-red-50 rounded-lg border border-red-200">
                      <div className="flex items-center justify-center gap-1 mb-1">
                        <TrendingUp className="w-4 h-4 text-red-600" />
                        <span className="text-xs font-semibold text-red-800">High</span>
                      </div>
                      <p className="text-xl font-bold text-red-600">
                        {extractedValues.filter(v => v.status === 'high').length}
                      </p>
                    </div>
                    <div className="text-center p-3 bg-orange-50 rounded-lg border border-orange-200">
                      <div className="flex items-center justify-center gap-1 mb-1">
                        <TrendingDown className="w-4 h-4 text-orange-600" />
                        <span className="text-xs font-semibold text-orange-800">Low</span>
                      </div>
                      <p className="text-xl font-bold text-orange-600">
                        {extractedValues.filter(v => v.status === 'low').length}
                      </p>
                    </div>
                  </div>
                </CardContent>
              )}
            </Card>
          )}



          {/* AI Report */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                AI-Generated Analysis
                {aiProvider && (
                  <span className="text-sm font-normal text-muted-foreground">({aiProvider})</span>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-6 space-y-6">
                {formatAnalysis(aiReport).map((item) => (
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
                
                {aiProvider?.includes('Template') && (
                  <div className="mt-4 pt-4 border-t border-blue-200">
                    <p className="text-xs text-gray-500 italic">
                      ℹ️ Template used. Configure API keys (GROQ_API_KEY, OPENAI_API_KEY, or HUGGINGFACE_API_KEY) in .env for AI-powered analysis.
                    </p>
                  </div>
                )}
              </div>
              
              <Button onClick={reset} variant="outline" className="mt-4">
                Reset & Analyze New Report
              </Button>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default MedicalReportUpload;