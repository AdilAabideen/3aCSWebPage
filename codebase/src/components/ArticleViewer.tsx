import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function ArticleViewer({content}) {



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

                <div className='w-full sm:w-[90%] md:w-[80%] aspect-video bg-muted rounded-lg overflow-hidden'>
                    <img src={content.image} alt={content.imageAlt} className="w-full h-full object-cover" />
                </div>

                <div className="prose prose-gray dark:prose-invert max-w-4xl sm:max-w-5xl mx-auto p-2 sm:p-4 pt-0 font-inter">
                    <ReactMarkdown 
                        remarkPlugins={[remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                            code({ node, inline, className, children, ...props }: { node: any, inline: any, className: any, children: any, props: any }) {
                                const match = /language-(\w+)/.exec(className || '');
                                return !inline && match ? (
                                  <SyntaxHighlighter
                                    style={vscDarkPlus}
                                    language={match[1]}
                                    PreTag="div"
                                    {...props}
                                  >
                                    {String(children).replace(/\n$/, '')}
                                  </SyntaxHighlighter>
                                ) : (
                                  <code className="bg-muted text-foreground px-1 sm:px-2 py-1 rounded text-sm sm:text-base" {...props}>
                                    {children}
                                  </code>
                                )
                            },
                            p: ({ node, ...props }) => <p className="font-inter text-base sm:text-lg font-light my-4 sm:my-6 leading-relaxed text-foreground" {...props} />,
                            h1: ({ node, ...props }) => <h1 className="font-medium text-2xl sm:text-3xl md:text-4xl my-4 sm:my-6 leading-tight text-foreground" {...props} />,
                            h2: ({ node, ...props }) => <h2 className="font-medium text-xl sm:text-2xl md:text-3xl my-4 sm:my-6 leading-tight text-foreground" {...props} />,
                            h3: ({ node, ...props }) => <h3 className="font-medium text-lg sm:text-xl md:text-2xl my-4 sm:my-6 leading-tight text-foreground" {...props} />,
                            h4: ({ node, ...props }) => <h4 className="font-medium text-base sm:text-lg md:text-xl my-3 sm:my-4 md:my-6 leading-tight text-foreground" {...props} />,
                            h5: ({ node, ...props }) => <h5 className="font-medium text-sm sm:text-base md:text-lg my-3 sm:my-4 leading-tight text-foreground" {...props} />,
                            h6: ({ node, ...props }) => <h6 className="font-medium text-xs sm:text-sm md:text-base my-2 sm:my-3 leading-tight text-foreground" {...props} />,
                            hr: ({ node, ...props }) => <hr className="my-6 sm:my-8 border-border" {...props} />,
                            ul: ({ node, ...props }) => <ul className="list-disc list-inside text-base sm:text-lg font-light my-2 sm:my-3 space-y-1 sm:space-y-2 text-foreground" {...props} />,
                            ol: ({ node, ...props }) => <ol className="list-decimal list-inside text-base sm:text-lg font-light my-2 sm:my-3 space-y-1 sm:space-y-2 text-foreground" {...props} />,
                            li: ({ node, ...props }) => <li className="font-inter text-base sm:text-lg font-light my-1 sm:my-2 leading-relaxed text-foreground" {...props} />,
                            strong: ({ node, ...props }) => <strong className="font-normal text-foreground" {...props} />,
                            em: ({ node, ...props }) => <em className="font-medium text-foreground" {...props} />,
                            blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-border pl-4 sm:pl-6 my-4 sm:my-6 italic text-base sm:text-lg text-foreground" {...props} />,
                            pre: ({ node, ...props }) => <pre className="bg-muted p-3 sm:p-4 rounded-lg overflow-x-auto text-sm sm:text-base my-4 sm:my-6 text-foreground" {...props} />,
                            table: ({ node, ...props }) => <table className="w-full border-collapse border border-border my-4 sm:my-6 text-sm sm:text-base" {...props} />,
                            th: ({ node, ...props }) => <th className="border border-border px-2 sm:px-3 py-2 text-left font-medium text-foreground" {...props} />,
                            td: ({ node, ...props }) => <td className="border border-border px-2 sm:px-3 py-2 text-foreground" {...props} />,
                          }}
                        
                    >{content.content}</ReactMarkdown>
                </div>

                
        </div>
      );
}
