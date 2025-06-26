import React from 'react';
import { ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';
import { articles } from '@/lib/articles';
const About = () => {
  const navigate = useNavigate()
  
  return (
    <section id="about" className="w-full py-20 px-6 md:px-12 bg-card relative overflow-hidden">
      {/* Background grid */}
      <div className="absolute inset-0 cosmic-grid opacity-20"></div>
      
      <div className="max-w-7xl mx-auto space-y-16 relative z-10">
        <div className="text-center space-y-4 max-w-3xl mx-auto">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-medium tracking-tighter text-foreground mb-4">
            About Us
          </h1>
          <h2 className="text-3xl md:text-4xl font-medium tracking-tighter text-foreground">
            Curated Tech • Creative Builds
          </h2>
          <p className="text-muted-foreground text-lg">
          We share what we find interesting — insightful articles, hands-on tutorials, and open source projects at the intersection of tech, AI, and creativity. No fluff, just useful ideas and experiments that inspire learning and building.
          </p>
        </div>
        
        <div className="mt-0">
            <div className="text-center mb-12">
                <h2 className="text-3xl md:text-4xl font-medium tracking-tighter text-foreground">
                    Trending
                </h2>
            </div>
            <div className="grid grid-cols-1  md:grid-cols-2 lg:grid-cols-3 gap-8">
            {articles.map((article, index) => (
              <div 
                key={index}
                onClick={() => navigate(`/article/${index}`)}
                className="p-6 m-0 rounded-xl border border-border bg-background/80 backdrop-blur-sm hover:border-border/60 transition-all duration-300 hover:scale-105 cursor-pointer flex flex-col justify-between"
              >
                <div>
                <div className="mb-6">
                  <h2 className="text-xl md:text-2xl font-medium text-foreground">{article.title}</h2>
                  <div className="flex flex-row text-sm text-muted-foreground gap-2 items-center">
                    <p>5 min read</p>
                    <p> • </p>
                    <span className="bg-blue-500 text-white px-2 py-1 rounded-lg text-xs font-medium">Tech</span>
                  </div>
                </div>
                <p className="text-lg mb-8 text-foreground/90 italic">"{article.cta}"</p>
                </div>
                <div className='flex flex-row w-full justify-between items-center'>
                  <div className="flex items-center gap-4">
                    <div className={`h-12 w-12 rounded-full bg-muted`}>
                      <img src={article.authorImage} alt={article.author} width={48} height={48} className="rounded-full" />
                    </div>
                    <div>
                      <h4 className="font-medium text-foreground">{article.author}</h4>
                      <p className="text-sm text-muted-foreground">{article.authorTitle}</p>
                    </div>
                  </div>
                  <ExternalLink size={26} className='inline-block cursor-pointer'/>
                </div>
              </div>
            ))}
            </div>
        <div className="text-center mt-12">
          <Button 
            className="px-8 py-3 text-xl font-medium bg-foreground text-background hover:bg-foreground/10 transition-colors duration-300 rounded-lg"
            onClick={() => navigate('/content')}
          >
            Read All Articles
            <ExternalLink size={24} className='inline-block cursor-pointer'/>
          </Button>
        </div>
        </div>
      </div>
    </section>
  );
};

export default About;
