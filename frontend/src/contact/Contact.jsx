import React from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faFacebookF, faInstagram, faLinkedinIn } from '@fortawesome/free-brands-svg-icons'

export default function Contact() {
  return (
    <div>
      <footer className="bg-[#455429] opacity-90 text-white py-8 px-4 mt-4" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.5)' }}>
        <div className="max-w-7xl mx-auto grid md:grid-cols-3 gap-6 text-center md:text-left">
          <div>
            <h2 className="text-lg font-semibold mb-2">Contact</h2>
            <p className="text-sm text-white">Email: raounak.hadil.kaoua@ensia.edu.dz</p>
            <p className="text-sm text-white">Phone: +213 794373602</p>
          </div>
            
            
          <div>
              <h2 className="text-lg font-semibold mb-2">Follow Us</h2>
              <div className="flex justify-center md:justify-start gap-4 space-x-4">
              <a href="https://www.facebook.com/share/16dTSAnATU/" target="_blank" rel="noopener noreferrer" className="text-white hover:text-white">
              <FontAwesomeIcon icon={faFacebookF} size="lg" />
              </a>
              <a href="https://www.instagram.com/5042ha?igsh=MTFsNG9vcjVhdWMzbA==" target="_blank" rel="noopener noreferrer" className="text-white hover:text-white">
              <FontAwesomeIcon icon={faInstagram} size="lg" />
              </a>
              <a href="https://www.linkedin.com/in/raounak-hadil-student-kaoua-a25505345?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" rel="noopener noreferrer" className="text-white hover:text-white">
              <FontAwesomeIcon icon={faLinkedinIn} size="lg" />
              </a>
            </div>
          </div>

        </div>
        <div className="mt-8 text-center text-white text-sm">
          &copy; 2025 Ai project. All rights reserved.
        </div>
      </footer>

    </div>
  )
}
