import { Brain, Shield, Zap, Users, FileText, Clock, CheckCircle, Star } from 'lucide-react';

const FeaturesSection = () => {
  const mainFeatures = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms trained on millions of medical reports provide accurate, instant insights.',
      benefits: ['99.9% accuracy rate', 'Instant processing', 'Continuous learning']
    },
    {
      icon: Shield,
      title: 'HIPAA Compliant Security',
      description: 'Enterprise-grade security ensures your medical data remains private and protected at all times.',
      benefits: ['End-to-end encryption', 'Secure data storage', 'Privacy by design']
    },
    {
      icon: Zap,
      title: 'Instant Results',
      description: 'Get comprehensive analysis and explanations of your medical reports in seconds, not days.',
      benefits: ['Real-time processing', '24/7 availability', 'No waiting times']
    },
    {
      icon: FileText,
      title: 'Multi-Format Support',
      description: 'Upload various types of medical reports including blood tests, CT scans, MRIs, and lab results.',
      benefits: ['PDF documents', 'Medical images', 'Lab reports']
    },
    {
      icon: Users,
      title: 'Doctor Collaboration',
      description: 'Share your analysis with healthcare providers for better communication and treatment planning.',
      benefits: ['Easy sharing', 'Professional reports', 'Better communication']
    },
    {
      icon: Clock,
      title: 'Historical Tracking',
      description: 'Track your health journey over time with comprehensive history and trend analysis.',
      benefits: ['Progress tracking', 'Trend analysis', 'Health insights']
    }
  ];

  const additionalFeatures = [
    'Plain English explanations',
    'Risk level assessments',
    'Personalized recommendations',
    'Mobile-friendly interface',
    'Multiple language support',
    'Export and sharing options',
    'Integration with health apps',
    'Regular AI model updates'
  ];

  return (
    <section id="features" className="py-20 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-bold font-heading mb-6">
            Powerful <span className="text-gradient-medical bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">Features</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Everything you need to understand your medical reports and take control of your health journey.
          </p>
        </div>

        {/* Main Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {mainFeatures.map((feature, index) => (
            <div 
              key={feature.title}
              className="glass-card p-8 hover:shadow-glow transition-all duration-300 animate-fade-in-up group"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {/* Icon */}
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-primary to-accent rounded-lg mb-6 group-hover:scale-110 transition-transform duration-300">
                <feature.icon className="h-8 w-8 text-white" />
              </div>

              {/* Content */}
              <h3 className="text-xl font-semibold font-heading mb-4">{feature.title}</h3>
              <p className="text-muted-foreground mb-6 leading-relaxed">
                {feature.description}
              </p>

              {/* Benefits */}
              <ul className="space-y-2">
                {feature.benefits.map((benefit, benefitIndex) => (
                  <li key={benefitIndex} className="flex items-center text-sm">
                    <CheckCircle className="h-4 w-4 text-accent mr-3 flex-shrink-0" />
                    <span className="text-muted-foreground">{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Additional Features */}
        <div className="glass-card p-8 animate-fade-in-up animate-delay-300">
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold font-heading mb-4">And Much More</h3>
            <p className="text-muted-foreground">
              MediScan comes packed with additional features to enhance your experience
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {additionalFeatures.map((feature, index) => (
              <div 
                key={feature}
                className="flex items-center space-x-3 p-4 bg-muted/50 rounded-lg hover:bg-muted transition-colors duration-200"
                style={{ animationDelay: `${index * 0.05}s` }}
              >
                <Star className="h-4 w-4 text-primary flex-shrink-0" />
                <span className="text-sm text-foreground">{feature}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-16 animate-fade-in-up animate-delay-400">
          <div className="glass-card p-8 max-w-3xl mx-auto">
            <h3 className="text-2xl font-bold font-heading mb-4">Ready to Experience MediScan?</h3>
            <p className="text-muted-foreground mb-6">
              Join thousands of users who trust MediScan for their medical report analysis. 
              Start your free trial today and see the difference AI can make.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="px-8 py-3 bg-gradient-to-r from-primary to-accent text-white rounded-lg font-medium hover:shadow-glow transition-all duration-300">
                Start Free Trial
              </button>
              <button className="px-8 py-3 border border-border text-foreground rounded-lg font-medium hover:bg-muted transition-all duration-300">
                Schedule Demo
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;