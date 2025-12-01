import React, { useState, useEffect } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

function App() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState(null);
  const [error, setError] = useState(null);
  const [steps, setSteps] = useState(12);  // Optimized default for faster generation
  const [guidance, setGuidance] = useState(7.5);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelStatus, setModelStatus] = useState('Initializing...');

  // Auto-load model on startup
  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      setModelStatus('Loading AI model...');
      const response = await fetch(`${API_URL}/model/load`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setModelStatus('Ready');
        setModelLoading(false);
      } else {
        setModelStatus('Model load failed');
        setModelLoading(false);
      }
    } catch (err) {
      setModelStatus('Backend not connected');
      setModelLoading(false);
    }
  };

  const presets = [
    { name: 'Modern Studio', prompt: 'architectural floor plan, modern studio apartment with open kitchen, detailed blueprint' },
    { name: 'Two Bedroom', prompt: 'architectural floor plan, two bedroom apartment with living room, professional drawing' },
    { name: 'Office Space', prompt: 'architectural floor plan, open office with meeting rooms, clean lines' },
    { name: 'Luxury Penthouse', prompt: 'architectural floor plan, luxury penthouse with master suite and balcony, detailed' },
    { name: 'Family Home', prompt: 'architectural floor plan, traditional family home with garage, professional' }
  ];

  const generateFloorPlan = async () => {
    if (!prompt.trim()) {
      setError('Please enter a description');
      return;
    }

    setLoading(true);
    setError(null);
    setImage(null);

    try {
      // Add architectural context to prompt
      const enhancedPrompt = prompt.includes('architectural') 
        ? prompt 
        : `architectural floor plan, ${prompt}, detailed blueprint, professional drawing`;

      const response = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          description: enhancedPrompt,
          steps: steps,
          guidance: guidance
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Generation failed');
      }

      const data = await response.json();
      
      if (data.status === 'success' && data.image) {
        setImage(data.image);
      } else {
        setError(data.error || 'Generation failed');
      }
    } catch (err) {
      setError(err.message || 'Failed to connect to backend');
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = () => {
    if (!image) return;
    
    const link = document.createElement('a');
    link.href = image;
    link.download = `floorplan-${Date.now()}.png`;
    link.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-6 max-w-7xl">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-slate-800 mb-2">
                FloorMind
              </h1>
              <p className="text-lg text-slate-600">
                AI-Powered Architectural Floor Plan Generator
              </p>
            </div>
            <div className="text-right">
              <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${
                modelLoading ? 'bg-yellow-100 text-yellow-800' : 
                modelStatus === 'Ready' ? 'bg-green-100 text-green-800' : 
                'bg-red-100 text-red-800'
              }`}>
                <span className={`w-2 h-2 rounded-full mr-2 ${
                  modelLoading ? 'bg-yellow-500 animate-pulse' : 
                  modelStatus === 'Ready' ? 'bg-green-500' : 
                  'bg-red-500'
                }`}></span>
                {modelStatus}
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main>
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Input Panel */}
            <div className="bg-white rounded-xl shadow-xl p-6 border border-slate-200">
              <h2 className="text-2xl font-semibold mb-6 text-slate-800 border-b pb-3">
                Configuration
              </h2>

              {/* Prompt Input */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Floor Plan Description
                </label>
                <textarea
                  className="w-full px-4 py-3 border-2 border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all resize-none"
                  rows="4"
                  placeholder="Describe your floor plan (e.g., modern 2 bedroom apartment with open kitchen and balcony)"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  disabled={modelLoading}
                />
              </div>

              {/* Presets */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-slate-700 mb-3">
                  Quick Templates
                </label>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {presets.map((preset, idx) => (
                    <button
                      key={idx}
                      onClick={() => setPrompt(preset.prompt)}
                      disabled={modelLoading}
                      className="px-3 py-2 bg-slate-100 text-slate-700 rounded-lg text-sm font-medium hover:bg-blue-100 hover:text-blue-700 transition-all border border-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {preset.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Parameters */}
              <div className="mb-6 space-y-4 bg-slate-50 p-4 rounded-lg border border-slate-200">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-sm font-semibold text-slate-700">
                      Quality (Steps)
                    </label>
                    <span className="text-sm font-mono text-slate-600 bg-white px-2 py-1 rounded border">
                      {steps}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="8"
                    max="30"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value))}
                    disabled={modelLoading}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>Faster (8-12 steps, ~30-60s)</span>
                    <span>Better Quality (20-30 steps, ~2-3min)</span>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-sm font-semibold text-slate-700">
                      Guidance Scale
                    </label>
                    <span className="text-sm font-mono text-slate-600 bg-white px-2 py-1 rounded border">
                      {guidance}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="12"
                    step="0.5"
                    value={guidance}
                    onChange={(e) => setGuidance(parseFloat(e.target.value))}
                    disabled={modelLoading}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>Creative</span>
                    <span>Accurate</span>
                  </div>
                </div>
              </div>

              {/* Generate Button */}
              <button
                onClick={generateFloorPlan}
                disabled={loading || !prompt.trim() || modelLoading}
                className={`w-full py-4 rounded-lg font-semibold text-white transition-all transform ${
                  loading || !prompt.trim() || modelLoading
                    ? 'bg-slate-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 hover:shadow-lg active:scale-95'
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Generating Floor Plan...
                  </span>
                ) : modelLoading ? (
                  'Loading Model...'
                ) : (
                  'Generate Floor Plan'
                )}
              </button>

              {/* Error Message */}
              {error && (
                <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded">
                  <p className="font-semibold">Error</p>
                  <p className="text-sm">{error}</p>
                </div>
              )}
            </div>

            {/* Output Panel */}
            <div className="bg-white rounded-xl shadow-xl p-6 border border-slate-200">
              <h2 className="text-2xl font-semibold mb-6 text-slate-800 border-b pb-3">
                Generated Result
              </h2>

              {image ? (
                <div className="space-y-4">
                  <div className="relative group">
                    <img
                      src={image}
                      alt="Generated floor plan"
                      className="w-full rounded-lg border-2 border-slate-200 shadow-md"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all rounded-lg"></div>
                  </div>
                  <button
                    onClick={downloadImage}
                    className="w-full py-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white rounded-lg font-semibold transition-all hover:shadow-lg active:scale-95"
                  >
                    Download Floor Plan
                  </button>
                </div>
              ) : (
                <div className="flex items-center justify-center min-h-[500px] bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg border-2 border-dashed border-slate-300">
                  <div className="text-center text-slate-500 p-8">
                    <svg className="mx-auto h-20 w-20 mb-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 5a1 1 0 011-1h4a1 1 0 010 2H6v13h13v-3a1 1 0 112 0v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 3h5v5M10 14l10-10" />
                    </svg>
                    <p className="text-lg font-medium text-slate-600 mb-2">No Floor Plan Generated Yet</p>
                    <p className="text-sm text-slate-500">Enter a description and click generate to create your floor plan</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Info Section */}
          <div className="mt-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
            <h3 className="text-lg font-semibold mb-4 text-slate-800">
              Tips for Best Results
            </h3>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-slate-700">
              <div className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Be specific about room types and quantities</span>
              </div>
              <div className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Mention architectural style preferences</span>
              </div>
              <div className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Include key features (balcony, garage, etc.)</span>
              </div>
              <div className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Use 8-12 steps for fast previews, 15-20 for final quality</span>
              </div>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="mt-8 text-center text-sm text-slate-500">
          <p>Powered by AI | Fine-tuned on 5,050 architectural floor plans</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
