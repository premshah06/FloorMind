import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Navbar from './components/layout/Navbar';
import Footer from './components/layout/Footer';
import HomePage from './pages/HomePage';
import GeneratorPage from './pages/GeneratorPage';
import ModelsPage from './pages/ModelsPage';
import MetricsPage from './pages/MetricsPage';
import AboutPage from './pages/AboutPage';
import DevelopersPage from './pages/DevelopersPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/generate" element={<GeneratorPage />} />
            <Route path="/models" element={<ModelsPage />} />
            {/* <Route path="/metrics" element={<MetricsPage />} /> */}
            <Route path="/about" element={<AboutPage />} />
            <Route path="/developers" element={<DevelopersPage />} />
          </Routes>
        </main>
        <Footer />
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1f2937',
              color: '#f9fafb',
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;