import React, { useState, useEffect } from 'react';

const Logo = () => {

  const [logoSrc, setLogoSrc] = useState("/images/3Alogo.png");

  useEffect(() => {
    const updateLogo = () => {
      const isDarkMode = document.documentElement.classList.contains('dark-mode');
      setLogoSrc(isDarkMode ? "/images/3ADarklogo.png" : "/images/3Alogo.png");
    };

    // Set the logo on initial render
    updateLogo();

    // Observe changes to the class attribute of the html element
    const observer = new MutationObserver(updateLogo);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

    // Clean up the observer on component unmount
    return () => observer.disconnect();
  }, []);

  return (
    <div className="flex items-center gap-2">
      <img 
        src={logoSrc} 
        alt={logoSrc} 
        className="h-12 w-12 object-contain"
      />
    </div>
  );
};

export default Logo;
