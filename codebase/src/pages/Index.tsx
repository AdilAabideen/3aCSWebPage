import React from 'react';
import Header from '@/components/Header';
import HeroSection from '@/components/HeroSection';
import Contact  from '@/components/Contact';
import Writers from '@/components/Writers';
import Footer from '@/components/Footer';
import About from '@/components/About';

const Index = () => {
  return (
    <div className=" bg-background text-foreground">
      <div className="relative min-h-[90vh]">
        <Header />
        <HeroSection />
      </div>
      <main>
        <About />
        <Writers />
        <Contact />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
