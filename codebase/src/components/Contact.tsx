import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';

const Contact = () => {
  return (
    <section id="contact" className="w-full py-20 px-6 md:px-12 bg-card">
      <div className="max-w-7xl mx-auto space-y-8 ">
        <div className="text-center space-y-4 max-w-3xl mx-auto ">
          <h2 className="text-4xl md:text-5xl font-medium tracking-tighter text-foreground">
            Contact Us
          </h2>
          <p className="text-muted-foreground text-lg">
            Work with us, Write with us, Build with us
          </p>
        </div>
        
        <form className="space-y-4 max-w-3xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1">
                    <Label htmlFor="name">Name</Label>
                    <Input id="name" placeholder="Enter your name" />
                </div>
                <div className="space-y-1">
                    <Label htmlFor="email">Email</Label>
                    <Input id="email" type="email" placeholder="Enter your email" />
                </div>
            </div>
            <div className="space-y-1">
                <Label htmlFor="subject">Subject</Label>
                <Input id="subject" placeholder="Enter the subject" />
            </div>
            <div className="space-y-1">
                <Label htmlFor="message">Message</Label>
                <Textarea id="message" placeholder="Enter your message" className="min-h-[120px]" />
            </div>
            <div>
                <Button type="submit" className="w-full">Send Message</Button>
            </div>
        </form>

        <div className="text-center text-muted-foreground">
          <p>
            Alternatively Email us at <a href="mailto:insight@3acs.xyz" className="text-primary hover:underline transition-all duration-300">insight@3acs.com</a>
          </p>
        </div>
      </div>
    </section>
  );
};

export default Contact;
