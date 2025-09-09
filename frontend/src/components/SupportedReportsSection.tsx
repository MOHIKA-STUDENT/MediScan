import { FileText, Heart, Microscope, Brain, Activity, Eye } from 'lucide-react';

const SupportedReportsSection = () => {
  const reportTypes = [
    {
      icon: Brain,
      title: "CT Scans",
      description: "Comprehensive analysis of CT scan images with AI-powered anomaly detection",
      features: ["Brain CT", "Chest CT", "Abdominal CT", "Full Body CT"],
      color: "from-primary to-primary-glow",
      available: true
    },
    {
      icon: Microscope,
      title: "Blood Tests",
      description: "Complete blood analysis including CBC, metabolic panels, and biomarkers",
      features: ["Complete Blood Count", "Lipid Panel", "Liver Function", "Kidney Function"],
      color: "from-medical-red to-medical-red/80",
      available: true
    },
    {
      icon: Heart,
      title: "ECG Reports",
      description: "Heart rhythm analysis and cardiovascular health assessment",
      features: ["Heart Rate Analysis", "Rhythm Detection", "Risk Assessment"],
      color: "from-accent to-mint",
      available: false
    },
    {
      icon: Eye,
      title: "MRI Scans",
      description: "Detailed MRI image analysis for various body systems",
      features: ["Brain MRI", "Spine MRI", "Joint MRI"],
      color: "from-purple-500 to-purple-700",
      available: false
    },
    {
      icon: Activity,
      title: "X-Ray Images",
      description: "Bone and tissue analysis from X-ray imaging",
      features: ["Chest X-Ray", "Bone X-Ray", "Joint X-Ray"],
      color: "from-green-500 to-green-700",
      available: false
    },
    {
      icon: FileText,
      title: "Lab Reports",
      description: "Comprehensive laboratory test result interpretation",
      features: ["Hormone Tests", "Vitamin Levels", "Allergy Tests"],
      color: "from-blue-500 to-blue-700",
      available: false
    }
  ];

  return (
    <section id="features" className="py-20 bg-muted/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Supported <span className="text-gradient-medical">Medical Reports</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Our AI can analyze various types of medical reports and imaging studies. 
            More report types are being added regularly.
          </p>
        </div>

        {/* Report Types Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {reportTypes.map((report, index) => (
            <div 
              key={report.title} 
              className={`glass-card p-6 transition-medical hover:shadow-glow animate-fade-in-up ${
                !report.available ? 'opacity-75' : ''
              }`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {/* Status Badge */}
              <div className="flex justify-between items-start mb-4">
                <div className={`p-3 rounded-lg bg-gradient-to-r ${report.color}`}>
                  <report.icon className="h-6 w-6 text-white" />
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                  report.available 
                    ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                    : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                }`}>
                  {report.available ? 'Available' : 'Coming Soon'}
                </div>
              </div>

              {/* Content */}
              <h3 className="text-xl font-semibold mb-3">{report.title}</h3>
              <p className="text-muted-foreground mb-4 leading-relaxed">
                {report.description}
              </p>

              {/* Features */}
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-foreground/80">Includes:</h4>
                <ul className="space-y-1">
                  {report.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-center text-sm text-muted-foreground">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full mr-3" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Action */}
              {report.available && (
                <div className="mt-6">
                  <button className="w-full btn-secondary-medical text-sm py-2">
                    Try Analysis
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Bottom Info */}
        <div className="mt-16 text-center animate-fade-in-up animate-delay-300">
          <div className="glass-card p-6 max-w-4xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
              <div>
                <div className="text-2xl font-bold text-primary mb-2">2+</div>
                <div className="text-muted-foreground">Report Types Available</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-accent-foreground mb-2">6+</div>
                <div className="text-muted-foreground">Total Types Planned</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-medical-red mb-2">24/7</div>
                <div className="text-muted-foreground">AI Analysis Available</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default SupportedReportsSection;