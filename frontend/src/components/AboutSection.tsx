import { Shield, Users, Award, Sparkles, Brain, Heart } from 'lucide-react';

const AboutSection = () => {
  const stats = [
    { number: '10,000+', label: 'Users Trusted', icon: Users },
    { number: '50,000+', label: 'Reports Analyzed', icon: Brain },
    { number: '99.9%', label: 'Accuracy Rate', icon: Award },
    { number: '24/7', label: 'AI Available', icon: Sparkles }
  ];

  const values = [
    {
      icon: Heart,
      title: 'Patient-Centered',
      description: 'We put patients first, making medical information accessible and understandable for everyone.'
    },
    {
      icon: Shield,
      title: 'Privacy & Security',
      description: 'HIPAA compliant infrastructure ensures your medical data remains private and secure.'
    },
    {
      icon: Brain,
      title: 'AI Innovation',
      description: 'Cutting-edge AI technology trained on millions of medical reports for accurate insights.'
    }
  ];

  return (
    <section id="about" className="py-20 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-bold font-heading mb-6">
            About <span className="text-gradient-medical bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">MediScan</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            We're revolutionizing healthcare accessibility by making medical reports understandable 
            for everyone through the power of artificial intelligence.
          </p>
        </div>

        {/* Mission Statement */}
        <div className="glass-card p-8 md:p-12 mb-16 animate-fade-in-up animate-delay-200">
          <div className="text-center max-w-4xl mx-auto">
            <h3 className="text-2xl md:text-3xl font-bold font-heading mb-6">Our Mission</h3>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Healthcare should be transparent and understandable. MediScan bridges the gap between 
              complex medical reports and patient understanding, empowering individuals to take 
              control of their health journey with confidence and clarity.
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
          {stats.map((stat, index) => (
            <div 
              key={stat.label} 
              className="text-center glass-card p-6 hover:shadow-glow transition-all duration-300 animate-fade-in-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-r from-primary to-accent rounded-lg mb-4">
                <stat.icon className="h-6 w-6 text-white" />
              </div>
              <div className="text-2xl md:text-3xl font-bold font-heading text-foreground mb-2">
                {stat.number}
              </div>
              <div className="text-sm text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>

        {/* Values */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          {values.map((value, index) => (
            <div 
              key={value.title} 
              className="text-center glass-card p-8 hover:shadow-glow transition-all duration-300 animate-fade-in-up"
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-primary to-accent rounded-lg mb-6">
                <value.icon className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold font-heading mb-4">{value.title}</h3>
              <p className="text-muted-foreground leading-relaxed">{value.description}</p>
            </div>
          ))}
        </div>

        {/* Team Section */}
        <div className="text-center animate-fade-in-up animate-delay-300">
          <div className="glass-card p-8 max-w-4xl mx-auto">
            <h3 className="text-2xl font-bold font-heading mb-6">Built by Healthcare & AI Experts</h3>
            <p className="text-muted-foreground mb-8 leading-relaxed">
              Our team combines decades of medical expertise with cutting-edge AI research. 
              We work closely with healthcare professionals to ensure our AI provides accurate, 
              reliable, and meaningful insights.
            </p>
            <div className="flex flex-wrap justify-center gap-6">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-primary rounded-full" />
                <span className="text-sm text-muted-foreground">Medical Doctors</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-accent rounded-full" />
                <span className="text-sm text-muted-foreground">AI Researchers</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-mint rounded-full" />
                <span className="text-sm text-muted-foreground">Healthcare IT Experts</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;