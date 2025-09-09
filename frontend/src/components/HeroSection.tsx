import { useState } from 'react';
import { ArrowRight, Upload, Brain, Shield, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';

const HeroSection = () => {
  const [isHovered, setIsHovered] = useState(false);

  const floatingElements = [
    { icon: Brain, delay: '0s', position: 'top-20 left-10' },
    { icon: Shield, delay: '2s', position: 'top-32 right-16' },
    { icon: Zap, delay: '4s', position: 'bottom-32 left-20' },
    { icon: Upload, delay: '1s', position: 'bottom-20 right-12' }
  ];

  return (
    <section id="home" className="relative min-h-screen flex items-center justify-center bg-gradient-hero overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0">
        {/* Floating Medical Icons */}
        {floatingElements.map((element, index) => (
          <div
            key={index}
            className={`absolute ${element.position} opacity-20`}
            style={{ animationDelay: element.delay }}
          >
            <element.icon className="h-16 w-16 text-primary medical-float" />
          </div>
        ))}
        
        {/* Gradient Orbs */}
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-primary/10 rounded-full blur-3xl medical-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl medical-pulse" style={{ animationDelay: '1s' }} />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="animate-fade-in-up">
          {/* Badge */}
          <div className="inline-flex items-center px-4 py-2 rounded-full glass-card mb-6">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2 medical-pulse" />
            <span className="text-sm font-medium text-foreground">
              AI-Powered Medical Analysis
            </span>
          </div>

          {/* Main Heading */}
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight">
            Revolutionizing{' '}
            <span className="text-gradient-medical">Health Reports</span>
            <br />
            with AI Intelligence
          </h1>

          {/* Subtitle */}
          <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
            Upload your CT Scans or Blood Reports and get instant, intelligent insights 
            that help you understand your health better than ever before.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12 animate-delay-200">
            <Button 
              className="btn-medical text-lg px-8 py-4 group"
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            >
              Get Started Free
              <ArrowRight className={`ml-2 h-5 w-5 transition-transform duration-300 ${
                isHovered ? 'translate-x-1' : ''
              }`} />
            </Button>
            <Button 
              variant="outline" 
              className="btn-secondary-medical text-lg px-8 py-4"
            >
              Watch Demo
            </Button>
          </div>

          {/* Trust Indicators */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-8 text-muted-foreground animate-delay-300">
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-green-500" />
              <span>HIPAA Compliant</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-primary" />
              <span>Instant Analysis</span>
            </div>
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-accent-foreground" />
              <span>AI-Powered</span>
            </div>
          </div>
        </div>

        {/* Medical Dashboard Preview */}
        <div className="mt-16 animate-fade-in-up animate-delay-300">
          <div className="glass-card p-8 max-w-4xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Upload Card */}
              <div className="glass-card p-6 text-center hover:shadow-glow transition-medical">
                <Upload className="h-12 w-12 text-primary mx-auto mb-4" />
                <h3 className="font-semibold mb-2">Upload Report</h3>
                <p className="text-sm text-muted-foreground">
                  Drag & drop your medical reports
                </p>
              </div>

              {/* Analysis Card */}
              <div className="glass-card p-6 text-center hover:shadow-glow transition-medical">
                <Brain className="h-12 w-12 text-accent-foreground mx-auto mb-4" />
                <h3 className="font-semibold mb-2">AI Analysis</h3>
                <p className="text-sm text-muted-foreground">
                  Get instant intelligent insights
                </p>
              </div>

              {/* Results Card */}
              <div className="glass-card p-6 text-center hover:shadow-glow transition-medical">
                <Shield className="h-12 w-12 text-green-500 mx-auto mb-4" />
                <h3 className="font-semibold mb-2">Secure Results</h3>
                <p className="text-sm text-muted-foreground">
                  View detailed health insights
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;