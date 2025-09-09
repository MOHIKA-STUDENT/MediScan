import MedVizioNavbar from '@/components/MedVizioNavbar';
import MedVizioHero from '@/components/MedVizioHero';
import FeaturesSection from '@/components/FeaturesSection';
import AboutSection from '@/components/AboutSection';
import ContactSection from '@/components/ContactSection';
import Footer from '@/components/Footer';

const MedVizioLanding = () => {
  return (
    <div className="min-h-screen">
      <MedVizioNavbar />
      <MedVizioHero />
      <FeaturesSection />
      <AboutSection />
      <ContactSection />
      <Footer />
    </div>
  );
};

export default MedVizioLanding;