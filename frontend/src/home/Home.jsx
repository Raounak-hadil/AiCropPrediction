import React from 'react'
import styles from './Home.module.css'
import Nav from "../nav/nav.jsx"
import { Link } from 'react-router-dom';
import Contact from '../contact/Contact.jsx'

export default function Home() {
  return (
    <div className={styles.main}>
      <div className="p-4">
        <Nav />
      </div>
      <div className="text-white flex flex-col justify-center items-center gap-10 p-12 h-screen" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.5)' }}>
          <h1 className="text-5xl text-center font-black md:text-7xl">Crop Prediction And Classification AI Agent</h1>
          <div className="text-2xl text-center md:text-4xl">Smart AI tool to help farmers and researchers predict <br></br>the best crops to grow and accurately classify crop types.</div>
          <Link to="/predict" className="text-xl bg-white text-black py-2 px-10 rounded-3xl flex justify-center cursor-pointer items-center gap-4 text-center">Predict now<div className="w-4 h-4 bg-[#1B5624] rounded-full text-center"></div></Link>
      </div>
      <div  id="contact">
        <Contact />
      </div>
    </div>
  )
}
