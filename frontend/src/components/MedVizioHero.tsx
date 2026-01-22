import { useState } from 'react';
import { ArrowRight, Upload, Brain, Shield, Sparkles, Play } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const MedVizioHero = () => {
  const [isHovered, setIsHovered] = useState(false);

  const floatingElements = [
    { icon: Brain, delay: '0s', position: 'top-20 left-10', size: 'h-12 w-12' },
    { icon: Shield, delay: '2s', position: 'top-32 right-16', size: 'h-10 w-10' },
    { icon: Sparkles, delay: '4s', position: 'bottom-32 left-20', size: 'h-8 w-8' },
    { icon: Upload, delay: '1s', position: 'bottom-20 right-12', size: 'h-14 w-14' }
  ];

  return (
    <section className="relative min-h-screen flex items-center justify-center bg-gradient-hero overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0">
        {/* Floating Medical Icons */}
        {floatingElements.map((element, index) => (
          <div
            key={index}
            className={`absolute ${element.position} opacity-10`}
            style={{ animationDelay: element.delay }}
          >
            <element.icon className={`${element.size} text-white medical-float`} />
          </div>
        ))}
        
        {/* Gradient Orbs */}
        <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-primary/20 rounded-full blur-3xl medical-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl medical-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-mint/15 rounded-full blur-3xl medical-pulse" style={{ animationDelay: '2s' }} />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="animate-fade-in-up">
    
        

          {/* Main Heading */}
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold font-heading mb-8 leading-tigh pt-10">
            Understand Your{' '}
       <span className="bg-gradient-to-r from-sky-600 via-cyan-600 to-emerald-500 bg-clip-text text-transparent">
  Medical Reports
</span>




            <br />
            Instantly with MediScan
          </h1>

          {/* Subtitle */}
          <p className="text-xl md:text-2xl text-gray-800 mb-12 max-w-4xl mx-auto leading-relaxed">
  Upload your health reports and get clear, personalized explanations within seconds. 
  No medical jargon, just insights you can understand and act upon.
</p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-16 animate-delay-200">
            <Link to="/signup">
              <Button 
                size="lg"
                className="text-lg px-10 py-6 bg-gradient-to-r from-primary to-accent text-white hover:shadow-glow transition-all duration-300 group"
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
              >
                Get Started Free
                <ArrowRight className={`ml-3 h-6 w-6 transition-transform duration-300 ${
                  isHovered ? 'translate-x-1' : ''
                }`} />
              </Button>
            </Link>
            <Button 
              variant="outline" 
              size="lg"
              className="text-lg px-10 py-6 bg-white/10 border-white/20 text-foreground hover:bg-white/20 transition-all duration-300"
            >
              <Play className="mr-3 h-5 w-5" />
              Watch Demo
            </Button>
          </div>

          {/* Trust Indicators */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-8  text-gray-800  animate-delay-300">
            <div className="flex items-center gap-3">
              <Shield className="h-6 w-6 text-accent" />
              <span className="font-medium">HIPAA Compliant & Secure</span>
            </div>
            <div className="flex items-center gap-3">
              <Sparkles className="h-6 w-6 text-primary" />
              <span className="font-medium">Instant AI Analysis</span>
            </div>
            <div className="flex items-center gap-3">
              <Brain className="h-6 w-6 text-mint" />
              <span className="font-medium">Trusted by Healthcare</span>
            </div>
          </div>
        </div>

        {/* Hero Dashboard Preview */}
        <div className="mt-16 mb-20 animate-fade-in-up animate-delay-300">
          <div className="glass-card p-8 max-w-5xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Upload Card */}
              <div className="glass-card p-8 text-center hover:shadow-glow transition-all duration-300 group">
                <div className="p-4 bg-gradient-to-r from-primary to-accent rounded-xl mx-auto mb-6 w-fit group-hover:scale-110 transition-transform duration-300">
                  <Upload className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold font-heading mb-3">Upload Report</h3>
                <p className="text-muted-foreground">
                  Drag & drop your medical reports or lab results securely
                </p>
              </div>

              {/* Analysis Card */}
              <div className="glass-card p-8 text-center hover:shadow-glow transition-all duration-300 group">
                <div className="p-4 bg-gradient-to-r from-accent to-mint rounded-xl mx-auto mb-6 w-fit group-hover:scale-110 transition-transform duration-300">
                  <Brain className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold font-heading mb-3">AI Analysis</h3>
                <p className="text-muted-foreground">
                  Get instant intelligent insights and explanations
                </p>
              </div>

              {/* Results Card */}
              <div className="glass-card p-8 text-center hover:shadow-glow transition-all duration-300 group">
                <div className="p-4 bg-gradient-to-r from-mint to-primary rounded-xl mx-auto mb-6 w-fit group-hover:scale-110 transition-transform duration-300">
                  <Shield className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold font-heading mb-3">Clear Results</h3>
                <p className="text-muted-foreground">
                  Understand your health with simple explanations
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default MedVizioHero;