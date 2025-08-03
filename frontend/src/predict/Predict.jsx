import styles from './Predict.module.css'
import Nav from '../nav/nav.jsx'
import React, { useState } from 'react';
import {useNavigate} from 'react-router-dom'
import axios from 'axios';

export default function Predict() {
    const [appear, setAppear] = useState(false);
    const [showAdvanced, setShowAdvanced] = useState(false); // <-- ADDED
    const [method, setMethod] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    function handleChoice(e) {
        e.preventDefault();
        appear ? setAppear(false) : setAppear(true);
    }

    function handleE(e) {
        if (["e", "E", "+", "-"].includes(e.key)) {
            e.preventDefault();
        }
    }

    function lastChoice(e) {
    e.preventDefault();
    const selectedMethod = e.target.value;
    setMethod(selectedMethod);
    setAppear(false)
    const form = document.querySelector("form");
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    data["pest_pressure"] = 10

    Object.keys(data).forEach((key) => {
        if (key !== "method") {
        data[key] = parseFloat(data[key]) || 0;
        }
    });

    data.method = selectedMethod;
    sendPredictionRequest(data);
    }

    const navigate = useNavigate();
    function greater(e) {
        if (+e.target.value > 300) e.target.value = 300; 
    }
    function sendPredictionRequest(formData) {
    setLoading(true); 

    axios.post('https://aicropprediction-6.onrender.com/api/predict/', formData)
        .then(response => {
            navigate('/result', { state: { result: response.data, userInputs: formData } });
        })
        .catch(error => {
            alert("Something went wrong. Please try again.");
            console.error("Error:", error);
        })
        .finally(() => {
            setLoading(false); 
        });
    }

        
  return (
    <div className={`p-4 ${styles.predict}`}>
        <div>
            <Nav />
        </div>
        <div className="flex justify-center items-center">
            <form className="relative bg-[rgba(250,255,241,0.9)] border-black text-[#455429] m-10 py-5 rounded-xl" action="" method="post">
                <h1 className="text-center text-3xl md:text-5xl pb-5 font-bold text-[#455429]">CROP PREDICTION</h1>
                <hr />
                <div className="p-10">
                    <div>
                        <h2 className="font-semibold text-2xl pb-2">Soil Characteristics</h2>
                        <hr className="text-[#B3D37A]"/>
                        <div className="flex flex-wrap flex-col justify-around items-center mt-3">
                        <div className="flex flex-wrap justify-around md:gap-10 items-center w-full">
                            <label className="pb-3" htmlFor="">Nitrogen (ppm):<br></br><input name="N" min={0} onInput={greater} onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                            <label className="pb-3" htmlFor="">Phosphorus (ppm):<br></br><input name="P" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                        </div>
                        <div className="flex flex-wrap justify-around md:gap-10 items-center w-full">
                            <label className="pb-3" htmlFor="">Potassium (ppm):<br></br><input name="K" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                            <label className="pb-3" htmlFor="">Soil Moisture (%):<br></br><input name="soil_moisture" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                        </div>
                        <div className="flex flex-wrap justify-around md:gap-10 items-center w-full">
                            <label className="pb-3" htmlFor="">Organic Matter (%):<br></br><input name="organic_matter" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                            <label className="pb-3" htmlFor="">Soil PH:<br></br><input name="ph" min={0} onInput={(e) => {  if (+e.target.value > 14) e.target.value = 14;  }}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                        </div>
                        <div className="flex flex-wrap justify-around md:gap-10 items-center w-full md:px-5">
                            <label className="pb-3 w-full" htmlFor="">
                                Soil Type:<br />
                                <select name="soil_type"  className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 pl-2">
                                <option name="1" value="1">Sandy</option>
                                <option name="2" value="2">Loamy</option>
                                <option name="3" value="3">Clay</option>
                                </select>
                            </label>
                        </div>
                        
                        </div>
                    </div>
                    <div>
                    <div>
                        <h2 className="font-semibold text-2xl pb-2">Climate Conditions</h2>
                        <hr className="text-[#B3D37A]"/>
                        <div className="flex flex-wrap flex-col justify-around items-center mt-3">
                        <div className="flex flex-wrap justify-around md:gap-10 items-center w-full">
                            <label className="pb-3" htmlFor="">Temperature (°C):<br></br><input name="temperature" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                            <label className="pb-3" htmlFor="">Humidity (%):<br></br><input name="humidity" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                        </div>
                        <div className="flex flex-wrap justify-around md:gap-10 items-center w-full">
                            <label className="pb-3" htmlFor="">Rainfall (mm):<br></br><input name="rainfall" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                            <label className="pb-3" htmlFor="">Sunlight Exposure (hrs/day):<br></br><input name="sunlight_exposure" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                        </div>
                        <div className="flex flex-wrap justify-around md:gap-10 items-center w-full">
                            <label className="pb-3" htmlFor="">Wind Speed (km/h):<br></br><input name="wind_speed" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                            <label className="pb-3" htmlFor="">CO₂ Concentration (ppm):<br></br><input name="co2_concentration" min={0} onInput={greater}  onKeyDown={handleE} required className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                        </div>
                        
                        </div>
                    </div>
                    </div>
                    <div>
                    <div>
                        <h2 className="font-semibold text-2xl pb-2">Other Conditions</h2>
                        <hr className="text-[#B3D37A]"/>
                        <div className="flex flex-wrap flex-col justify-around items-center mt-3">
                        <div className="flex flex-wrap justify-around md:gap-20 items-center w-full">
                            <label className="pb-3" htmlFor="">Irrigation Frequency (ti/we):<br></br><input name="irrigation_frequency" min={0} onInput={greater}  onKeyDown={handleE} className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                            <label className="pb-3" htmlFor="">Fertilizer Usage (kg/ha):<br></br><input name="fertilizer_usage" min={0} onInput={greater}  onKeyDown={handleE} className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                        </div>
                        <div className="flex flex-wrap justify-around md:gap-20 items-center w-full">
                            <label className="pb-3" htmlFor="">Urban Area Proximity (km):<br></br><input name="urban_area_proximity" min={0} onInput={greater}  onKeyDown={handleE} className="border border-[#455429] bg-white outline-none rounded-2xl my-5 w-full h-10 p-4" type="number" /></label>
                            <label className="pb-3" htmlFor="">
                                Water Source Type:<br />
                                <select name="water_source_type" className="border border-[#455429] bg-white outline-none rounded-2xl my-5 h-10 w-53 px-4 appearance-none">
                                <option value="1">River</option>
                                <option value="2">Groundwater</option>
                                <option value="3">Recycled</option>
                                </select>
                            </label>
                        </div>
                        
                        </div>
                    </div>
                    </div>
                    <hr className="text-[#B3D37A]"/>
                    <div  className="flex justify-center items-center mt-10">
                        <button onClick={handleChoice} className="bg-[#455429] px-10 py-3 rounded-xl text-white text-xl" type="submit">Predict</button>
                    </div>
                </div>  
            </form>
            {/* MODAL UPDATED */}
            <div className={`${styles.appear} ${appear ? '' : 'hidden'} fixed top-24 left-1/2 transform -translate-x-1/2 z-50 w-[80%] max-w-4xl p-10 border border-white bg-[rgba(250,255,241,0.95)] rounded-xl`}>
                {!showAdvanced ? (
                    <div className="flex flex-wrap justify-center items-center gap-5">
                        <button onClick={lastChoice} value="csp" disabled={loading} className="bg-[#B3D37A] text-[#455429] py-3 px-7 rounded-xl" id="csp">CSP</button>
                        <button onClick={() => setShowAdvanced(true)} className="bg-[#FFB347] text-[#455429] py-3 px-7 rounded-xl">
                            Advanced Settings
                        </button>
                    </div>
                ) : (
                    <div className="flex flex-col justify-center items-center gap-5">
                        <p className="text-red-600 text-center mb-4 font-semibold">
                            ⚠ Warning: These methods can take a long time !
                        </p>
                        <div className="flex flex-wrap justify-center items-center gap-5 pb-4">
                            <button onClick={lastChoice} value="astar" disabled={loading} className="bg-[#B3D37A] text-[#455429] py-3 px-7 rounded-xl" id="a_star">A star</button>
                            <button onClick={lastChoice} value="greedy" disabled={loading} className="bg-[#B3D37A] text-[#455429] py-3 px-7 rounded-xl" id="greedy">Greedy</button>
                            <button onClick={lastChoice} value="genetic" disabled={loading} className="bg-[#B3D37A] text-[#455429] py-3 px-7 rounded-xl" id="genetic">Genetic</button>
                            <button 
                                onClick={() => setShowAdvanced(false)} 
                                className="bg-[#B3D37A] text-[#455429] py-3 px-7 rounded-xl">
                                ⬅ Back
                            </button>
                        </div>

                    </div>
                )}
            </div>
        </div>
        {loading && (
        <div className="fixed inset-0 bg-opacity-40 flex justify-center items-center z-50">
            <div className="bg-white text-[#455429] px-6 py-4 rounded-xl shadow-xl text-xl font-semibold">
            ⏳ Please wait, it could take a while ...
            </div>
        </div>
        )}
        {error && <div className="text-red-500">{error}</div>}
    </div>
  )
}
