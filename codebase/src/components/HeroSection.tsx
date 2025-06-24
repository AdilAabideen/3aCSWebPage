import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import TaskBoard from './TaskBoard';
import { Loader } from 'lucide-react';
import { Button } from './ui/button';
import Logo from './Logo';
import { toast } from 'sonner';

const HeroSection = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // States for the email input
  const[ email, setEmail ] = useState('');
  const[ isSubscribing, setIsSubscribing ] = useState(false);
  const [subscribedSuccess, setSubscribedSuccess] = useState(false);
 

  useEffect(() => {
    // Simulate loading time
    const loadingTimer = setTimeout(() => {
      setIsLoading(false);
    }, 1500);

    // Show content after loading
    const contentTimer = setTimeout(() => {
      setIsVisible(true);
    }, 1800);

    return () => {
      clearTimeout(loadingTimer);
      clearTimeout(contentTimer);
    };
  }, []);
  const handleSubscribe = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) {
      toast.error('Please enter your email address');
      return;
    }

    // Email validation regex
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      toast.error('Please enter a valid email address');
      return;
    }
    setIsSubscribing(true);
    try {
      console.log('Subscribing to newsletter...');
    const response = await fetch(`/api/subscribe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email }),
    });

    if (!response.ok) {
      throw new Error('Subscription failed');
    }

    const data = await response.json();
    toast.success('Successfully subscribed to newsletter!');
    setEmail('');
    setSubscribedSuccess(true);
    } catch (error) {
      console.error('Error subscribing to newsletter:', error);
      toast.error('Failed to subscribe. Please try again.');
    } finally {
      setIsSubscribing(false);
      
    }
  }

  if (isLoading) {
    return (
      <section id="home" className="absolute top-0 left-0 right-0 w-full pt-0 pb-20 md:pb-32 px-6 md:px-12 flex flex-col items-center justify-center overflow-hidden bg-background min-h-screen">
        <div className="flex flex-col items-center justify-center space-y-6">
          <Loader className="h-12 w-12 animate-spin text-primary" />
          <p className="text-lg text-muted-foreground animate-pulse">Loading...</p>
        </div>
      </section>
    );
  }

  return (
    <section id="home" className="absolute top-0 left-0 right-0 w-full pt-0 pb-20 md:pb-32 px-6 md:px-12 flex flex-col items-center justify-center overflow-hidden bg-background min-h-screen">
      {/* Cosmic particle effect (background dots) */}
      <div className="absolute inset-0 cosmic-grid opacity-30"></div>
      
      {/* Gradient glow effect */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full">
        <div className="w-full h-full opacity-10 bg-primary blur-[120px]"></div>
      </div>
      
      <div className={`relative z-10 max-w-4xl text-center space-y-6 transition-all duration-1000 transform pt-20 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
        <div className="flex justify-center mb-4">
          <div className="scale-150 md:scale-[2.5]">
            <Logo />
          </div>
        </div>
        <h1 className="text-4xl md:text-6xl lg:text-7xl font-medium tracking-tighter text-balance text-foreground">
          <span className="text-5xl md:text-7xl lg:text-8xl">Creative Curiosity</span> <span className="text-primary">Intelligent</span> Futures
        </h1>
        
        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto text-balance">
          A space for exploring how AI and emerging tech are shaping creativity, thought, and innovation. From tutorials to trends, we share ideas that spark discovery and build what's next.
        </p>
        
        <div className="pt-6 min-h-[100px]">
          {subscribedSuccess ? (
            <div className="flex flex-col items-center justify-center text-center animate-in fade-in duration-500">
              <h3 className="text-2xl font-medium text-primary">Thank you for subscribing!</h3>
              <p className="text-muted-foreground mt-2">You're on the list. We'll be in touch soon.</p>
            </div>
          ) : (
            <form onSubmit={handleSubscribe} className="flex flex-col sm:flex-row justify-center items-center gap-2 sm:gap-0 animate-in fade-in duration-300">
              <Input 
                type="email" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isSubscribing}
                placeholder="Enter your email address" 
                className="w-full max-w-md h-16 text-lg md:text-lg rounded-xl sm:rounded-r-none focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-2 focus-visible:border-primary"
              />
              <Button 
                type="submit"
                disabled = {isSubscribing}
                className="w-full sm:w-auto h-16 text-xl rounded-xl sm:rounded-l-none focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-2 focus-visible:border-primary">
                  {isSubscribing ? <Loader className="h-6 w-6 animate-spin" /> : 'Subscribe'}
                </Button>
            </form>
          )}
        </div>
        
        <div className="pt-6 text-sm text-muted-foreground">
          Enter your email to get the latest updates • No credit card required 
        </div>
        
      </div>
    </section>
  );
};

export default HeroSection;
