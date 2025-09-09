import { Upload, Brain, FileText, ArrowRight } from 'lucide-react';

const HowItWorksSection = () => {
  const steps = [
    {
      step: 1,
      icon: Upload,
      title: "Upload Your Report",
      description: "Simply drag and drop your CT scan or blood test report. We support multiple formats including PDF, JPEG, and PNG.",
      color: "text-primary"
    },
    {
      step: 2,
      icon: Brain,
      title: "AI Analysis",
      description: "Our advanced AI algorithms analyze your medical data in seconds, identifying key patterns and anomalies.",
      color: "text-accent-foreground"
    },
    {
      step: 3,
      icon: FileText,
      title: "Get Insights",
      description: "Receive a comprehensive, easy-to-understand report with visual insights and recommended next steps.",
      color: "text-medical-red"
    }
  ];

  return (
    <section id="how-it-works" className="py-20 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            How <span className="text-gradient-medical">MediScan</span> Works
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Get professional medical insights in three simple steps. Our AI-powered platform 
            makes understanding your health reports easier than ever.
          </p>
        </div>

        {/* Steps */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
          {steps.map((stepItem, index) => (
            <div key={stepItem.step} className="relative">
              {/* Step Card */}
              <div className="glass-card p-8 text-center hover:shadow-glow transition-medical animate-fade-in-up" style={{ animationDelay: `${index * 0.2}s` }}>
                {/* Step Number */}
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="w-8 h-8 bg-gradient-to-r from-primary to-primary-glow rounded-full flex items-center justify-center text-primary-foreground font-bold text-sm">
                    {stepItem.step}
                  </div>
                </div>

                {/* Icon */}
                <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-primary/10 to-accent/10 mb-6 mt-4`}>
                  <stepItem.icon className={`h-8 w-8 ${stepItem.color}`} />
                </div>

                {/* Content */}
                <h3 className="text-xl font-semibold mb-4">{stepItem.title}</h3>
                <p className="text-muted-foreground leading-relaxed">
                  {stepItem.description}
                </p>
              </div>

              {/* Arrow (except for last step) */}
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                  <ArrowRight className="h-8 w-8 text-accent-foreground opacity-30" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-16 animate-fade-in-up animate-delay-300">
          <div className="glass-card p-8 max-w-2xl mx-auto">
            <h3 className="text-2xl font-semibold mb-4">Ready to analyze your reports?</h3>
            <p className="text-muted-foreground mb-6">
              Join thousands of users who trust MediScan for their health insights.
            </p>
            <button className="btn-medical">
              Start Your Free Analysis
              <ArrowRight className="ml-2 h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorksSection;