
import React from 'react';
import { Linkedin, Github } from 'lucide-react';
const Writers = () => {
  const testimonials = [
    {
      quote: "Our payment processing efficiency increased by 40% and transaction failures dropped to near zero. The automation features are game-changing.",
      author: "Sarah Johnson",
      position: "CFO at TechCorp",
      avatar: "bg-cosmic-light/30"
    },
    {
      quote: "The real-time analytics and fraud detection capabilities have saved us millions. We can spot issues before they become problems.",
      author: "Michael Chen",
      position: "Head of Risk at FinanceFlow",
      avatar: "bg-cosmic-light/20"
    },
    {
      quote: "Compliance used to be a nightmare. Now our regulatory reporting is automated and we're always audit-ready.",
      author: "Leila Rodriguez",
      position: "Operations Director at GlobalPay",
      avatar: "bg-cosmic-light/40"
    }
  ];
  
  return (
    <section className="w-full py-20 px-6 md:px-12  relative overflow-hidden" id="creators">
      {/* Background grid */}
      <div className="absolute inset-0 cosmic-grid opacity-20"></div>
      
      <div className="max-w-7xl mx-auto space-y-16 relative z-10">
        <div className="text-center space-y-4 max-w-3xl mx-auto">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium tracking-tighter text-foreground mb-4">
            Creators
          </h1>
          <h2 className="text-3xl md:text-4xl font-medium tracking-tighter text-foreground">
            Minds behind the Content
          </h2> 
          <p className="text-muted-foreground text-lg">
          Articles, tutorials, projects, and more written by a growing group of curious builders, thinkers, and tech explorers.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3  gap-8">
        <div 
              className=" hover:scale-103 m-0 p-8 rounded-xl border border-border bg-background/80 backdrop-blur-sm hover:border-border/60 transition-all duration-300"
            >
              <div className="mb-6  flex flex-row justify-between items-center">
                <div>
                  <h2 className="text-2xl md:text-3xl font-medium text-foreground ">Adil Aabideen</h2>
                  <p className="text-sm text-muted-foreground">Lead Writer</p>
                </div>
                <div className={`h-12 w-12 rounded-full bg-muted`}>
                  <img src="/images/adilProfile.jpeg" alt="Adil Aabideen" width={48} height={48} className="rounded-full" />
                </div>
              </div>
              <p className="text-lg mb-8 text-foreground/90 italic">"This is the Software Century. The next 100 years will be defined by the development and integration of software and technology into every part of life. The only question is: will you help shape it — or be left behind?"</p>
              <div className="flex items-center gap-4  ">
                <a href="https://www.linkedin.com/in/adil-aabideen/" target="_blank" rel="noopener noreferrer">
                  <Linkedin size={24} className='inline-block cursor-pointer transition-all duration-300 hover:scale-125'/>
                </a>
                <a href="https://github.com/AdilAabideen" target="_blank" rel="noopener noreferrer">
                  <Github size={24} className='inline-block cursor-pointer transition-all duration-300 hover:scale-125'/>
                </a>
              </div>
            </div>
        </div>
      </div>
    </section>
  );
};

export default Writers;
