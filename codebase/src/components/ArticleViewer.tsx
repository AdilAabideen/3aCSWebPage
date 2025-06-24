import React from 'react'
import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import content1 from '../md/article_1.md?raw'

export default function ArticleViewer({content}) {
    const [contentArticle, setContentArticles] = useState('')

    useEffect(() => {
        import(/* @vite-ignore */ `../md/${content.content}?raw`).then((md) => {
            setContentArticles(md.default);
          });
    }, [content.content])

    return (
        <div className='bg-inherit max-w-6xl mx-auto flex flex-col gap-6 sm:gap-8 md:gap-12 items-center justify-center my-6 sm:my-8 md:my-10 py-4 px-4 sm:px-6'>
            <div className='w-full'>
                    <div className="mb-4 sm:mb-6 flex flex-col gap-2">
                        <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-medium text-foreground leading-tight">{content.title}</h2>
                        <div className="flex flex-col sm:flex-row text-sm text-muted-foreground gap-1 sm:gap-2 items-start sm:items-center">
                            <p>5 min read</p>
                            <p className="hidden sm:inline"> • </p>
                            <span className="bg-blue-500 text-white px-2 py-1 rounded-lg text-xs font-medium w-fit">Tech</span>
                            <p className="hidden sm:inline"> • </p>
                            <p>A 3A Article</p>
                        </div>
                    </div>
                    <p className="text-base sm:text-lg mb-6 sm:mb-8 text-foreground/90 italic leading-relaxed">"{content.cta}"</p>
                    <div className='flex flex-col sm:flex-row w-full justify-between items-start sm:items-center gap-3 sm:gap-0'>
                        <a href={content.authorLink} target='_blank' className="w-full sm:w-auto">
                            <div className="flex items-center gap-3 sm:gap-4 hover:cursor-pointer hover:scale-105 transition-all duration-300">
                                <div className={`h-10 w-10 sm:h-12 sm:w-12 rounded-full bg-muted flex-shrink-0`}>
                                    <img src={content.authorImage} alt={content.author} width={48} height={48} className="rounded-full w-full h-full object-cover" />
                                </div>
                                <div className="min-w-0 flex-1">
                                    <h4 className="font-medium text-foreground text-sm sm:text-base truncate">{content.author}</h4>
                                    <p className="text-xs sm:text-sm text-muted-foreground truncate">{content.authorTitle}</p>
                                </div>
                            </div>
                        </a>
                    </div>
                </div>

                <div className='w-full sm:w-[90%] md:w-[80%] aspect-video bg-gray-300 rounded-lg overflow-hidden'>
                    <img src={content.image} alt={content.imageAlt} className="w-full h-full object-cover" />
                </div>

                <div className="prose max-w-4xl sm:max-w-5xl mx-auto p-2 sm:p-4 pt-0 font-inter">
                    <ReactMarkdown 
                        components={{
                            p: ({ node, ...props }) => <p className="font-inter text-base sm:text-lg font-light my-4 sm:my-6 leading-relaxed" {...props} />,
                            h1: ({ node, ...props }) => <h1 className="font-medium text-2xl sm:text-3xl md:text-4xl my-4 sm:my-6 leading-tight" {...props} />,
                            h2: ({ node, ...props }) => <h2 className="font-medium text-xl sm:text-2xl md:text-3xl my-4 sm:my-6 leading-tight" {...props} />,
                            h3: ({ node, ...props }) => <h3 className="font-medium text-lg sm:text-xl md:text-2xl my-4 sm:my-6 leading-tight" {...props} />,
                            h4: ({ node, ...props }) => <h4 className="font-medium text-base sm:text-lg md:text-xl my-3 sm:my-4 md:my-6 leading-tight" {...props} />,
                            h5: ({ node, ...props }) => <h5 className="font-medium text-sm sm:text-base md:text-lg my-3 sm:my-4 leading-tight" {...props} />,
                            h6: ({ node, ...props }) => <h6 className="font-medium text-xs sm:text-sm md:text-base my-2 sm:my-3 leading-tight" {...props} />,
                            hr: ({ node, ...props }) => <hr className="my-6 sm:my-8" {...props} />,
                            ul: ({ node, ...props }) => <ul className="list-disc list-inside text-base sm:text-lg font-light my-2 sm:my-3 space-y-1 sm:space-y-2" {...props} />,
                            ol: ({ node, ...props }) => <ol className="list-decimal list-inside text-base sm:text-lg font-light my-2 sm:my-3 space-y-1 sm:space-y-2" {...props} />,
                            li: ({ node, ...props }) => <li className="font-inter text-base sm:text-lg font-light my-1 sm:my-2 leading-relaxed" {...props} />,
                            strong: ({ node, ...props }) => <strong className="font-normal" {...props} />,
                            em: ({ node, ...props }) => <em className="font-medium" {...props} />,
                            blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-gray-300 pl-4 sm:pl-6 my-4 sm:my-6 italic text-base sm:text-lg" {...props} />,
                            code: ({ node, ...props }) => <code className="bg-gray-100 dark:bg-gray-800 px-1 sm:px-2 py-1 rounded text-sm sm:text-base" {...props} />,
                            pre: ({ node, ...props }) => <pre className="bg-gray-100 dark:bg-gray-800 p-3 sm:p-4 rounded-lg overflow-x-auto text-sm sm:text-base my-4 sm:my-6" {...props} />,
                            table: ({ node, ...props }) => <table className="w-full border-collapse border border-gray-300 my-4 sm:my-6 text-sm sm:text-base" {...props} />,
                            th: ({ node, ...props }) => <th className="border border-gray-300 px-2 sm:px-3 py-2 text-left font-medium" {...props} />,
                            td: ({ node, ...props }) => <td className="border border-gray-300 px-2 sm:px-3 py-2" {...props} />,
                          }}
                    >{contentArticle}</ReactMarkdown>
                </div>

                
        </div>
      );
}
