import React from 'react'
import Header from '@/components/Header'
import Articles from '@/components/Articles'

export default function Content() {
  return (
    <div className="bg-background text-foreground">
        <div className='relative '>
            <Header />
        </div>
        <main>
            <Articles />
        </main>
    </div>
  )
}
