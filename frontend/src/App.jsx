import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Home from './home/Home'
import Predict from './predict/Predict';
import Result from './result/Result'
import About from './about/About'

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="predict" element={<Predict />} />
        <Route path="result" element={<Result />} />
        <Route path="about" element={<About />} />
      </Routes>
    </Router>
  );
}
