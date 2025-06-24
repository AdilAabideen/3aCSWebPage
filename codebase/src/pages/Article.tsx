import React from 'react'
import ArticleViewer from '@/components/ArticleViewer'
import Header from '@/components/Header'
import Footer from '@/components/Footer'
import { articles } from '@/lib/articles'
import { useParams } from 'react-router-dom'

export default function Article() {
  const { slug } = useParams()
  const article = articles[slug]

  return (
    <div className='bg-background text-foreground'>
        <div className='relative'>
            <Header />
        </div>
        <ArticleViewer content={article} />
        <Footer />
    </div>
  )
}

