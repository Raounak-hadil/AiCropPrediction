import React from 'react'
import { Link } from 'react-router-dom';
import { useState } from 'react';

export default function Nav() {
  const [burger, Setburger] = useState();
  function handleBurger(e) {
    burger ? Setburger(false) : Setburger(true);
    console.log("well")
  }
  return (
    <div style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.5)' }}>
      <div onClick={handleBurger} className={`${burger ? 'hidden' : ''} space-y-1 cursor-pointer md:hidden`}>
        <span className="block w-8 h-1 bg-white rounded"></span>
        <span className="block w-8 h-1 bg-white rounded"></span>
        <span className="block w-8 h-1 bg-white rounded"></span>
      </div>
      <div onClick={handleBurger} className={`${burger ? '' : 'hidden'} fixed top-0 left-0 z-50 bg-[#455429] opacity-90 w-full h-70 p-4`}>
        <div>
          <div className="relative w-8 h-8" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }}>
            <span className="absolute top-1/2 left-0 w-5 h-1 bg-white transform rotate-45"></span>
            <span className="absolute top-1/2 left-0 w-5 h-1 bg-white transform -rotate-45"></span>
          </div>
          <ul className="flex flex-col justify-center items-center text-white gap-8">
            <Link to="/" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }}>Home</Link>
            <Link to="/about" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }}>AboutUs</Link>
          <Link 
              to="/#contact" 
              style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }}
              onClick={() => window.location.pathname === '/' && 
                document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' })} >
              ContactUs
            </Link>
            <Link to="/predict" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }} className="bg-white rounded-xl text-[#1B5624] px-5 py-1">Predict</Link>
          </ul>
        </div>
      </div>
      <div className="hidden md:flex justify-between items-center text-white px-5 text-2xl font-bold">
          <Link to="/" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }}>AiPredict</Link>
          <div>
            <ul className="flex justify-center items-center text-white gap-8">
              <Link to="/" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }}>Home</Link>
              <Link to="/about" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }}>AboutUs</Link>
              <Link 
                  to="/#contact" 
                  style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }}
                  onClick={() => window.location.pathname === '/' && 
                    document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' })} >
                  ContactUs
                </Link>
              <Link to="/predict" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0)' }} className="bg-white rounded-xl text-[#1B5624] px-5 py-1">Predict</Link>
            </ul>
          </div>
      </div>
    </div>
  )
}
