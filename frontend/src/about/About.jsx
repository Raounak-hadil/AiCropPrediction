import React from 'react'
import styles from './About.module.css'
import Nav from '../nav/nav.jsx'

export default function About() {
  return (
    <div className={`${styles.about} p-4 text-white`}>
      <Nav />
      <div className="p-8 max-w-4xl mx-auto text-center space-y-6 text-white" style={{ textShadow: '2px 2px 4px rgba(0,0,0,1)' }}>
        <h1 className="text-7xl font-bold">About Us</h1>
        <p className=" text-2xl">
          Welcome to <span className="font-semibold">AgroPredict</span> – a smart solution designed to help farmers and agricultural experts make data-driven decisions.
        </p>
        <p className=" text-2xl">
          Our mission is to empower sustainable farming by combining the power of Artificial Intelligence and real-world data. With our crop prediction model, users can anticipate the best crops to grow based on soil, weather, and environmental conditions.
        </p>
        <p className=" text-2xl">
          This project was developed by passionate students from [Your University or Team Name], driven by a desire to bring tech to the heart of agriculture.
        </p>
        <h2 className="text-6xl font-semibold">Our Vision</h2>
        <p className=" text-2xl">
          To revolutionize agriculture in developing regions with accessible and intelligent farming tools.
        </p>
      </div>
    </div>
  )
}
