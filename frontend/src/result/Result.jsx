import React from 'react'
import Nav from '../nav/nav'
import styles from './Result.module.css';
import { useState } from 'react';
import  { Link } from 'react-router-dom'
import { useLocation } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faXmark } from '@fortawesome/free-solid-svg-icons';

export default function Result() {
  const [appear, setAppear] = useState(false);
  const location = useLocation();
  const result = location.state?.result;

  function handleXmark(e) {
    appear ? setAppear(false) : setAppear(true);
  }

  function handleChoice(e) {
      e.preventDefault();
      appear ? setAppear(false) : setAppear(true);
  }

  const cropImages = {
    orange: "/crops/orange.jpg",
    apple: "/crops/apple.jpg",
    banana: "/crops/banana.jpg",
    blackgram: "/crops/blackgram.jpg",
    rice: "/crops/rice.jpg",
    maize: "/crops/maize.jpg",
    chickpea: "/crops/chickpea.jpg",
    kidneybeans: "/crops/kidneybeans.jpg",
    pigeonpeas: "/crops/pigeonpeas.jpg",
    mothbeans: "/crops/mothbeans.jpg",
    mungbean: "/crops/mungbean.jpg",
    blackgram: "/crops/blackgram.jpg",
    lentil: "/crops/lentil.jpg",
    pomegranate: "/crops/pomegranate.jpg",
    mango: "/crops/mango.jpg",
    grapes: "/crops/grapes.jpg",
    watermelon: "/crops/watermelon.jpg",
    muskmelon: "/crops/muskmelon.jpg",
    papaya: "/crops/papaya.jpg",
    coconut: "/crops/coconut.jpg",
    cotton: "/crops/cotton.jpg",
    jute: "/crops/jute.jpg",
    coffee: "/crops/coffee.jpg",
  };

  return (
    <div className={`${styles.result} p-4`}>
          <div>
            <Nav />
        </div>
        <div className="flex justify-center items-center">
            <div className="relative bg-[rgba(250,255,241,0.9)]  border-black text-[#B3D37A] m-10 py-5 rounded-xl">
                <h1 className="text-center text-3xl md:text-5xl pb-5 font-bold text-[#455429]">Best Crop: <br></br> </h1>
                <hr />
                <div className="my-4 flex flex-wrap flex-row justify-evenly items-center">
                  <div className="flex justify-center items-center px-4">
                    <img className="rounded-xl object-cover w-90 h-60" src={cropImages[result?.bestCrop.name]} alt=""/>
                  </div>
                  <div className="text-center text-[#455429]">
                    <div className="text-4xl p-4">{result?.bestCrop.name}</div>
                    <div>Growth Stage: {result?.bestCrop.growthStage}</div>
                    <div className="rounded-full bg-[#4554295c] p-8 text-2xl m-4"> {result?.bestCrop.matchingPercentage}%<br/><span className="text-xl">matching</span></div>
                  </div>
                </div>
                <hr />
                <div className="p-10 text-[#455429]">
                    <div className="font-semibold text-2xl pb-2">Envioronment Condition</div>
                    <hr className="text-[#B3D37A]"></hr>
                    <div className="flex flex-wrap justify-start items-center pt-8 sm:gap-20 md:mx-10 md:mb-20">
                      <div className="flex flex-wrap flex-col justify-start">
                      <h2 className="font-semibold text-2xl pb-2">Yours</h2>
                      <hr className="text-[#B3D37A]"/>
                      <div className="flex flex-wrap flex-col justify-around items-center mt-3">
                        <ul className="flex flex-wrap flex-col justify-start items-start w-full">
                          <li className="pb-3">Nitrogen (ppm): {result?.userCondition.N}</li>
                          <li className="pb-3">Phosphorus (ppm): {result?.userCondition.P}</li>
                          <li className="pb-3">Potassium (ppm): {result?.userCondition.K}</li>
                          <li className="pb-3">Soil Moisture (%): {result?.userCondition.soil_moisture}</li>
                          <li className="pb-3">Organic Matter (%): {result?.userCondition.organic_matter}</li>
                          <li className="pb-3">Soil pH: {result?.userCondition.ph}</li>
                          <li className="pb-3">Soil Type: {result?.userCondition.soil_type}</li>
                          <li className="pb-3">Temperature (°C): {result?.userCondition.temperature}</li>
                          <li className="pb-3">Humidity (%) {result?.userCondition.humidity}</li>
                          <li className="pb-3">Rainfall (mm): {result?.userCondition.rainfall}</li>
                          <li className="pb-3">Sunlight Exposure (hrs/day): {result?.userCondition.sunlight_exposure}</li>
                          <li className="pb-3">Wind Speed (km/h): {result?.userCondition.wind_speed}</li>
                          <li className="pb-3">CO₂ Concentration (ppm): {result?.userCondition.co2_concentration}</li>
                          <li className="pb-3">Urban Area Proximity (km): {result?.userCondition.urban_area_proximity}</li>
                          <li className="pb-3">Water Source Type: {result?.userCondition.water_source_type}</li>
                        </ul>
                      </div>
                      </div>
                      <div className="md:w-px md:h-screen md:bg-[#B3D37A] md:mx-4"></div>
                      <div>
                        <h2 className="font-semibold text-2xl pb-2">Crop</h2>
                        <hr className="text-[#B3D37A]"/>
                        <div className="flex flex-wrap flex-col justify-around items-center mt-3">
                          <ul className="flex flex-wrap flex-col justify-start items-start w-full">
                            <li className="pb-3">Nitrogen (ppm): {result?.environmentCondition.N}</li>
                            <li className="pb-3">Phosphorus (ppm):  {result?.environmentCondition.P}</li>
                            <li className="pb-3">Potassium (ppm): {result?.environmentCondition.K}</li>
                            <li className="pb-3">Soil Moisture (%): {result?.environmentCondition.soil_moisture}</li>
                            <li className="pb-3">Organic Matter (%): {result?.environmentCondition.organic_matter}</li>
                            <li className="pb-3">Soil pH: {result?.environmentCondition.ph}</li>
                            <li className="pb-3">Soil Type: {result?.environmentCondition.soil_type}</li>
                            <li className="pb-3">Temperature (°C): {result?.environmentCondition.temperature}</li>
                            <li className="pb-3">Humidity (%) {result?.environmentCondition.humidity}</li>
                            <li className="pb-3">Rainfall (mm): {result?.environmentCondition.rainfall}</li>
                            <li className="pb-3">Sunlight Exposure (hrs/day): {result?.environmentCondition.sunlight_exposure}</li>
                            <li className="pb-3">Wind Speed (km/h): {result?.environmentCondition.wind_speed}</li>
                            <li className="pb-3">CO₂ Concentration (ppm): {result?.environmentCondition.co2_concentration}</li>
                            <li className="pb-3">Urban Area Proximity (km): {result?.environmentCondition.urban_area_proximity}</li>
                            <li className="pb-3">Water Source Type: {result?.environmentCondition.water_source_type}</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                    {/*Done */}
                    <div>
                      <div>
                          <h2 className="font-semibold text-2xl pb-2">Recommended Crop Management:</h2>
                          <hr className="text-[#B3D37A]"/>
                          <div className="flex flex-wrap flex-col justify-start items-start mt-3 md:px-10 py-5">
                            <ul className="flex flex-wrap flex-col justify-start items-start w-full">
                              <li className="pb-3">Frost Risk (index): 90.86</li>
                              <li className="pb-3">Irrigation Frequency (times/week): 5</li>
                              <li className="pb-3">Crop Density (plants/m²): 7.41</li>
                              <li className="pb-3">Pest Pressure (index): 69.62</li>
                              <li className="pb-3">Fertilizer Usage (kg/ha): 125.66</li>
                              <li className="pb-3">Water Usage Efficiency (L/kg): 3.77</li>
                            </ul>
                          </div>
                      </div>
                    </div>
                    <hr className="text-[#B3D37A]"/>
                    <div  className="flex flex-wrap gap-5 md:gap-20 justify-center items-center mt-10">
                        <button className="bg-[#455429] px-10 py-3 rounded-xl text-white text-xl" type="submit"><Link to="/">Go Back Home</Link></button>
                        <button onClick={handleChoice} className="bg-[#455429] px-10 py-3 rounded-xl text-white text-xl" type="submit">See Other Crop</button>
                    </div>
                </div>  
            </div>
            
            <div className={`${styles.appear} ${appear ? '' : 'hidden'} fixed top-32 left-1/2 transform -translate-x-1/2 z-50 w-[80%] sm:w-[60%] md:w-[40%] lg:w-[30%] p-4 border border-white bg-[rgba(250,255,241,0.95)] rounded-xl`}>
                <div className="flex flex-wrap flex-col gap-5">
                    <div onClick={handleXmark}><FontAwesomeIcon icon={faXmark} /></div>
                    {result?.otherCrops.map((crop, index) => (
                    <div key={index} className="bg-[#B3D37A] text-[#455429] py-3 px-7 rounded-xl flex items-center gap-5">
                      <div>
                        <div className="text-xl font-bold">{crop.name}</div>
                        <div>{crop.matchingPercentage}% matching</div>
                      </div>
                    </div>
                  ))}
                </div>
            </div>
        </div>
    </div>
  )
}
