import { ExternalLink } from 'lucide-react'
import React, { useState, useEffect } from 'react'
import {articles} from '@/lib/articles'
import { useNavigate } from 'react-router-dom'
export default function Articles() {
  const [isVisible, setIsVisible] = useState(false);
  const navigate = useNavigate()
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true);
    }, 300); // 300ms delay

    return () => clearTimeout(timer);
  }, []);



  return (
    <section className='w-full py-6 sm:py-8 md:py-10 px-4 sm:px-6 md:px-12 relative overflow-hidden gap-4 sm:gap-6 flex flex-col' id='articles'>
        <div className='max-w-7xl space-y-8 sm:space-y-12 md:space-y-16 relative z-10'>
            <div className='text-left space-y-3 sm:space-y-4 max-w-3xl mx-auto'>
                <h1 className={`text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-medium tracking-tighter text-foreground mb-2 sm:mb-4 transition-all duration-700 transform ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
                    Articles
                </h1>
            </div>
        </div>

        <div className={`grid grid-cols-1 gap-4 sm:gap-6 md:gap-8 max-w-6xl mx-auto transition-all duration-1000 transform ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            {articles.map((article, index) => (
                <div onClick={() => navigate(`/article/${index}`)} key={index} className="cursor-pointer p-4 sm:p-6 m-0 grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-0 rounded-xl border border-border bg-background/80 backdrop-blur-sm hover:border-border/60 transition-all duration-300 hover:scale-[1.02] sm:hover:scale-105">
                    <div className='col-span-1 lg:col-span-2 order-2 lg:order-1'>
                        <div className="mb-4 sm:mb-6 flex flex-col gap-2">
                            <h2 className="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-medium text-foreground leading-tight">{article.title}</h2>
                            <div className="flex flex-col sm:flex-row text-sm text-muted-foreground gap-1 sm:gap-2 items-start sm:items-center">
                                <p>5 min read</p>
                                <p className="hidden sm:inline"> • </p>
                                <span className="bg-blue-500 text-white px-2 py-1 rounded-lg text-xs font-medium w-fit">Tech</span>
                            </div>
                        </div>
                        <p className="text-base sm:text-lg mb-6 sm:mb-8 text-foreground/90 italic leading-relaxed">"{article.cta}"</p>
                        <div className='flex flex-col sm:flex-row w-full justify-between items-start sm:items-center gap-3 sm:gap-0'>
                            <a href={article.authorLink} target='_blank' className="w-full sm:w-auto">
                                <div className="flex items-center gap-3 sm:gap-4">
                                    <div className={`h-10 w-10 sm:h-12 sm:w-12 rounded-full bg-muted flex-shrink-0`}>
                                        <img src={article.authorImage} alt={article.author} width={48} height={48} className="rounded-full w-full h-full object-cover" />
                                    </div>
                                    <div className="min-w-0 flex-1">
                                        <h4 className="font-medium text-foreground text-sm sm:text-base truncate">{article.author}</h4>
                                        <p className="text-xs sm:text-sm text-muted-foreground truncate">{article.authorTitle}</p>
                                    </div>
                                </div>
                            </a>
                            <ExternalLink size={20} className='sm:hidden inline-block cursor-pointer text-muted-foreground'/>
                            <ExternalLink size={26} className='hidden sm:inline-block cursor-pointer text-muted-foreground'/>
                        </div>
                    </div>
                    <div className='col-span-1 order-1 md:order-2'>
                        <div className=" lg:w-full flex justify-center items-center w-full rounded-lg overflow-hidden">
                            <img 
                              src={article.image} 
                              alt={article.imageAlt} 
                              className=" w-full lg:w-full  p-4 mt-2  lg:h-56 object-cover rounded-xl"
                            />
                        </div>
                    </div>
                </div>
            ))}
        </div>
    </section>
  )
}
