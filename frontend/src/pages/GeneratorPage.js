import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Settings2, Download, RefreshCw, Wand2, Clock, Image as ImageIcon, AlertCircle, History, ChevronDown, ChevronUp, Building2 } from 'lucide-react';
import toast from 'react-hot-toast';
import floorMindAPI from '../services/api';
import { useModel } from '../context/ModelContext';

const GeneratorPage = () => {
  // Use global model state
  const { modelLoaded, modelInfo, setModelLoaded, setModelInfo } = useModel();
  // Load persisted state from localStorage
  const loadPersistedState = () => {
    try {
      const saved = localStorage.getItem('generatorPageState');
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (error) {
      console.error('Error loading persisted state:', error);
    }
    return null;
  };

  const persistedState = loadPersistedState();

  const [prompt, setPrompt] = useState(persistedState?.prompt || '');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState(persistedState?.generatedImage || null);
  const [generationMetrics, setGenerationMetrics] = useState(persistedState?.generationMetrics || null);
  const [style, setStyle] = useState(persistedState?.style || 'modern');
  const [apiStatus, setApiStatus] = useState('checking'); // checking, online, offline, loading_model
  const [presets, setPresets] = useState(null);
  const [showFamousPlans, setShowFamousPlans] = useState(persistedState?.showFamousPlans || false);
  const [estimatedTime, setEstimatedTime] = useState(8); // Default 8 seconds

  // Persist state to localStorage whenever it changes (excluding modelLoaded/modelInfo - now in global context)
  useEffect(() => {
    const stateToSave = {
      prompt,
      generatedImage,
      generationMetrics,
      style,
      showFamousPlans
    };
    localStorage.setItem('generatorPageState', JSON.stringify(stateToSave));
  }, [prompt, generatedImage, generationMetrics, style, showFamousPlans]);

  // Estimate generation time based on device
  useEffect(() => {
    if (modelInfo) {
      const device = modelInfo.device || 'cpu';
      // Estimate based on device type
      if (device === 'cuda') {
        setEstimatedTime(5); // GPU: ~3-5s, use 5s for progress bar
      } else if (device === 'mps') {
        setEstimatedTime(8); // MPS: ~5-8s, use 8s for progress bar
      } else {
        setEstimatedTime(90); // CPU: ~1-2min, use 90s for progress bar
      }
    }
  }, [modelInfo]);

  // Initialize API and load presets
  useEffect(() => {
    const initializeAPI = async () => {
      try {
        // Check API health
        await floorMindAPI.checkHealth();
        setApiStatus('online');
        
        // Only show success toast if model wasn't already loaded
        if (!modelLoaded) {
          toast.success('FloorMind AI is ready!');
        }

        // Load model info - always check with API for fresh status
        try {
          const info = await floorMindAPI.getModelInfo();
          const isLoaded = info.model_info?.is_loaded || false;
          
          // Update global state with fresh data from API
          setModelInfo(info.model_info);
          setModelLoaded(isLoaded);
          
          // Show toast if model status changed
          if (isLoaded && !modelLoaded) {
            toast.success('Model is loaded and ready!');
          }
        } catch (error) {
          console.warn('Could not load model info:', error);
        }

        // Load presets
        try {
          const presetsData = await floorMindAPI.getPresets();
          setPresets(presetsData.presets);
        } catch (error) {
          console.warn('Could not load presets:', error);
        }

      } catch (error) {
        setApiStatus('offline');
        toast.error('FloorMind AI is offline. Please start the backend server.');
        console.error('API initialization failed:', error);
      }
    };

    initializeAPI();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Get sample prompts from presets or use defaults
  const samplePrompts = presets?.residential || [
    "3-bedroom apartment with open kitchen and living room",
    "Small studio with bathroom and kitchenette", 
    "2-story house with 4 bedrooms and 2 bathrooms",
    "Modern loft with master bedroom and walk-in closet",
    "Family home with garage and dining room",
    "Office space with conference room and reception area"
  ];

  // Famous floor plans that users can select
  const famousFloorPlans = [
    {
      category: "Residential Classics",
      plans: [
        {
          name: "Victorian Townhouse",
          description: "Victorian era townhouse with parlor, dining room, kitchen, 3 bedrooms, library, and servants quarters",
          style: "traditional",
          image: "🏛️"
        },
        {
          name: "Mid-Century Modern Ranch",
          description: "1950s mid-century modern ranch house with open floor plan, 3 bedrooms, 2 bathrooms, and carport",
          style: "contemporary",
          image: "🏡"
        },
        {
          name: "Brownstone Apartment",
          description: "Classic New York brownstone apartment with railroad layout, 2 bedrooms, galley kitchen, and bay windows",
          style: "traditional",
          image: "🏢"
        },
        {
          name: "Craftsman Bungalow",
          description: "Early 1900s craftsman bungalow with front porch, living room with fireplace, 2 bedrooms, and built-in cabinetry",
          style: "traditional",
          image: "🏠"
        }
      ]
    },
    {
      category: "Modern Luxury",
      plans: [
        {
          name: "Penthouse Suite",
          description: "Luxury penthouse with open concept living, 4 bedrooms, master suite with spa bathroom, chef's kitchen, and wraparound terrace",
          style: "modern",
          image: "🌆"
        },
        {
          name: "Contemporary Villa",
          description: "Modern villa with infinity pool, 5 bedrooms, home theater, wine cellar, and floor-to-ceiling windows",
          style: "modern",
          image: "🏰"
        },
        {
          name: "Minimalist Loft",
          description: "Industrial minimalist loft with exposed brick, open living space, mezzanine bedroom, and concrete floors",
          style: "minimalist",
          image: "🏭"
        }
      ]
    },
    {
      category: "Commercial Spaces",
      plans: [
        {
          name: "Boutique Hotel Suite",
          description: "Boutique hotel suite with king bedroom, sitting area, kitchenette, luxury bathroom with soaking tub",
          style: "contemporary",
          image: "🏨"
        },
        {
          name: "Modern Office Layout",
          description: "Open office floor plan with hot desks, meeting rooms, break room, private offices, and collaboration zones",
          style: "modern",
          image: "🏢"
        },
        {
          name: "Restaurant Floor Plan",
          description: "Restaurant layout with dining area, bar, kitchen, prep area, storage, and restrooms",
          style: "contemporary",
          image: "🍽️"
        },
        {
          name: "Retail Store",
          description: "Retail store with display areas, fitting rooms, checkout counter, storage room, and customer lounge",
          style: "modern",
          image: "🛍️"
        }
      ]
    },
    {
      category: "Unique Designs",
      plans: [
        {
          name: "Japanese Tea House",
          description: "Traditional Japanese tea house with tatami rooms, engawa veranda, tokonoma alcove, and zen garden view",
          style: "traditional",
          image: "🎋"
        },
        {
          name: "Tiny House",
          description: "Efficient tiny house with loft bedroom, compact kitchen, fold-down dining, and multi-purpose living area",
          style: "minimalist",
          image: "🏘️"
        },
        {
          name: "Artist Studio",
          description: "Artist studio with high ceilings, north-facing windows, work area, storage for supplies, and small living quarters",
          style: "industrial",
          image: "🎨"
        }
      ]
    }
  ];

  const handleSelectFamousPlan = (plan) => {
    setPrompt(plan.description);
    setStyle(plan.style);
    setShowFamousPlans(false);
    toast.success(`Selected: ${plan.name}`);
  };

  const handleClearAll = () => {
    setPrompt('');
    setGeneratedImage(null);
    setGenerationMetrics(null);
    setStyle('modern');
    setShowFamousPlans(false);
    localStorage.removeItem('generatorPageState');
    toast.success('Cleared all inputs and results');
  };

  const handleLoadModel = async () => {
    if (apiStatus !== 'online') {
      toast.error('Backend server is not available');
      return;
    }

    setApiStatus('loading_model');
    
    try {
      toast.loading('Loading FloorMind AI model...', { duration: 2000 });
      
      const result = await floorMindAPI.loadModel();
      
      if (result.status === 'success') {
        setModelLoaded(true);
        setModelInfo(result.model_info);
        toast.success('🎉 FloorMind AI model loaded successfully!');
      } else {
        throw new Error(result.error || 'Failed to load model');
      }
      
    } catch (error) {
      console.error('Model loading error:', error);
      toast.error(`Failed to load model: ${error.message}`);
      setModelLoaded(false);
    } finally {
      setApiStatus('online');
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a description for your floor plan');
      return;
    }

    if (apiStatus !== 'online') {
      toast.error('FloorMind AI is not available. Please check the backend server.');
      return;
    }

    if (!modelLoaded) {
      toast.error('Please load the FloorMind AI model first.');
      return;
    }

    setIsGenerating(true);
    const startTime = Date.now();
    
    try {
      // Call your trained FloorMind SDXL model
      // Using 10 steps for ultra-fast generation (3-5 seconds on GPU!)
      const result = await floorMindAPI.generateFloorPlan({
        description: prompt,
        style: style,
        width: 512,
        height: 512,
        steps: 10,  // Ultra-fast mode: 10 steps (~3-5s on GPU)
        guidance: 7.5,
        save: true
      });
      
      // Save to history
      if (result.success) {
        const historyItem = {
          id: Date.now(),
          prompt: prompt,
          style: style,
          image: result.image,
          timestamp: new Date().toISOString(),
          metadata: result.metadata
        };
        
        // Get existing history from localStorage
        const existingHistory = JSON.parse(localStorage.getItem('floorplanHistory') || '[]');
        existingHistory.unshift(historyItem); // Add to beginning
        
        // Keep only last 150 items
        const trimmedHistory = existingHistory.slice(0, 150);
        localStorage.setItem('floorplanHistory', JSON.stringify(trimmedHistory));
      }

      if (result.success) {
        setGeneratedImage(result.image);
        setGenerationMetrics({
          generation_time: (Date.now() - startTime) / 1000,
          metadata: result.metadata
        });
        
        toast.success('🎉 Floor plan generated successfully! Saved to history.');
      } else {
        throw new Error('Generation failed');
      }
      
    } catch (error) {
      console.error('Generation error:', error);
      toast.error(`Generation failed: ${error.message}`);
      
      // Reset states on error
      setGeneratedImage(null);
      setGenerationMetrics(null);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSamplePrompt = (samplePrompt) => {
    setPrompt(samplePrompt);
  };

  const handleDownload = async () => {
    if (generatedImage) {
      try {
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        const filename = `floor_plan_${timestamp}.png`;
        
        await floorMindAPI.downloadImage(generatedImage, filename);
        toast.success('🎉 Floor plan downloaded successfully!');
      } catch (error) {
        console.error('Download error:', error);
        toast.error('Download failed. Please try again.');
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
            <span className="gradient-text">AI Floor Plan</span> Generator
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Transform your ideas into detailed architectural floor plans using your trained AI model
          </p>
          
          {/* API Status Indicator */}
          <div className="mt-6 flex justify-center">
            <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${
              (apiStatus === 'online' && modelLoaded)
                ? 'bg-green-100 text-green-800' 
                : apiStatus === 'offline'
                ? 'bg-red-100 text-red-800'
                : 'bg-yellow-100 text-yellow-800'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                (apiStatus === 'online' && modelLoaded)
                  ? 'bg-green-500' 
                  : apiStatus === 'offline'
                  ? 'bg-red-500'
                  : 'bg-yellow-500 animate-pulse'
              }`} />
              {(apiStatus === 'online' && modelLoaded) && 'FloorMind AI Ready'}
              {(apiStatus === 'online' && !modelLoaded) && 'Model Not Loaded'}
              {apiStatus === 'offline' && 'FloorMind AI Offline'}
              {apiStatus === 'checking' && 'Connecting to FloorMind AI...'}
              {apiStatus === 'loading_model' && 'Loading AI Model...'}
            </div>
            
            {/* Model Load Button */}
            {apiStatus === 'online' && !modelLoaded && (
              <button
                onClick={handleLoadModel}
                className="ml-4 inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-full text-sm font-medium hover:bg-primary-700 transition-colors"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Load AI Model
              </button>
            )}
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Prompt Input */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-semibold text-gray-900">
                  Describe Your Floor Plan
                </label>
                {(prompt || generatedImage) && (
                  <button
                    onClick={handleClearAll}
                    className="text-xs px-3 py-1 bg-red-50 text-red-600 rounded-full hover:bg-red-100 transition-colors flex items-center space-x-1"
                    disabled={isGenerating}
                  >
                    <RefreshCw className="w-3 h-3" />
                    <span>Clear All</span>
                  </button>
                )}
              </div>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter a detailed description of your desired floor plan..."
                className="w-full h-32 p-4 border border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                disabled={isGenerating}
              />
              
              {/* Sample Prompts */}
              <div className="mt-4">
                <p className="text-sm font-medium text-gray-700 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {samplePrompts.slice(0, 3).map((sample, index) => (
                    <button
                      key={index}
                      onClick={() => handleSamplePrompt(sample)}
                      className="text-xs px-3 py-1 bg-primary-50 text-primary-700 rounded-full hover:bg-primary-100 transition-colors"
                      disabled={isGenerating}
                    >
                      {sample}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Famous Floor Plans - Collapsible */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <button
                onClick={() => setShowFamousPlans(!showFamousPlans)}
                className="w-full flex items-center justify-between text-left"
                disabled={isGenerating}
              >
                <div className="flex items-center space-x-2">
                  <Building2 className="w-5 h-5 text-primary-600" />
                  <span className="text-sm font-semibold text-gray-900">Famous Floor Plans</span>
                  <span className="text-xs px-2 py-1 bg-primary-100 text-primary-700 rounded-full">
                    {famousFloorPlans.reduce((acc, cat) => acc + cat.plans.length, 0)} plans
                  </span>
                </div>
                {showFamousPlans ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </button>

              <AnimatePresence>
                {showFamousPlans && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden"
                  >
                    <div className="mt-4 space-y-4 max-h-96 overflow-y-auto">
                      {famousFloorPlans.map((category, catIndex) => (
                        <div key={catIndex}>
                          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                            {category.category}
                          </h4>
                          <div className="space-y-2">
                            {category.plans.map((plan, planIndex) => (
                              <motion.button
                                key={planIndex}
                                onClick={() => handleSelectFamousPlan(plan)}
                                disabled={isGenerating}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                className="w-full text-left p-3 bg-gradient-to-r from-gray-50 to-gray-100 hover:from-primary-50 hover:to-secondary-50 rounded-lg border border-gray-200 hover:border-primary-300 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                <div className="flex items-start space-x-3">
                                  <span className="text-2xl">{plan.image}</span>
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between mb-1">
                                      <p className="text-sm font-medium text-gray-900 truncate">
                                        {plan.name}
                                      </p>
                                      <span className="text-xs px-2 py-0.5 bg-white border border-gray-200 text-gray-600 rounded-full capitalize ml-2">
                                        {plan.style}
                                      </span>
                                    </div>
                                    <p className="text-xs text-gray-600 line-clamp-2">
                                      {plan.description}
                                    </p>
                                  </div>
                                </div>
                              </motion.button>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Model Info */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <label className="block text-sm font-semibold text-gray-900 mb-3">
                <Settings2 className="w-4 h-4 inline mr-2" />
                AI Model Information
              </label>
              <div className="space-y-3">
                <div className="flex items-center space-x-3 p-3 bg-gradient-to-r from-primary-50 to-secondary-50 rounded-lg border border-primary-100">
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center">
                      <Sparkles className="w-5 h-5 text-white" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">
                      FloorMind SDXL Model
                      {apiStatus === 'online' && modelLoaded && (
                        <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">
                          ✓ Active
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-gray-600">
                      Fine-tuned Stable Diffusion XL for floor plans
                    </div>
                    {modelInfo && (
                      <div className="text-xs text-gray-500 mt-1">
                        Type: {modelInfo.model_type || 'SDXL'} • Resolution: {modelInfo.resolution || '512-1024'}px • Device: {modelInfo.device || 'Auto'}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Style and 3D Options */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <label className="block text-sm font-semibold text-gray-900 mb-3">
                <Wand2 className="w-4 h-4 inline mr-2" />
                Advanced Options
              </label>
              
              <div className="space-y-4">
                {/* Style Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Architectural Style
                  </label>
                  <select
                    value={style}
                    onChange={(e) => setStyle(e.target.value)}
                    disabled={isGenerating}
                    className="w-full p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  >
                    <option value="modern">Modern</option>
                    <option value="contemporary">Contemporary</option>
                    <option value="traditional">Traditional</option>
                    <option value="minimalist">Minimalist</option>
                    <option value="industrial">Industrial</option>
                    <option value="scandinavian">Scandinavian</option>
                  </select>
                </div>


              </div>
            </div>

            {/* Generate Button */}
            <motion.button
              onClick={handleGenerate}
              disabled={isGenerating || !prompt.trim() || !modelLoaded || apiStatus !== 'online'}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full bg-gradient-to-r from-primary-600 to-secondary-600 text-white font-semibold py-4 px-6 rounded-xl hover:from-primary-700 hover:to-secondary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl disabled:hover:scale-100"
            >
              {isGenerating ? (
                <div className="flex items-center justify-center space-x-2">
                  <motion.div 
                    className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                  <span>Generating Floor Plan...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <Sparkles className="w-5 h-5" />
                  <span>Generate Floor Plan</span>
                </div>
              )}
            </motion.button>
          </motion.div>

          {/* Output Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Generated Image */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Generated Floor Plan</h3>
                {generatedImage && (
                  <div className="flex gap-2">
                    <Link
                      to="/history"
                      className="flex items-center space-x-2 px-4 py-2 bg-secondary-100 text-secondary-700 rounded-lg hover:bg-secondary-200 transition-colors"
                    >
                      <History className="w-4 h-4" />
                      <span>View History</span>
                    </Link>
                    <button
                      onClick={handleDownload}
                      className="flex items-center space-x-2 px-4 py-2 bg-primary-100 text-primary-700 rounded-lg hover:bg-primary-200 transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download</span>
                    </button>
                  </div>
                )}
              </div>
              
              <div className="aspect-square bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl border-2 border-dashed border-gray-200 flex items-center justify-center relative overflow-hidden">
                {isGenerating ? (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center relative"
                  >
                    {/* Modern 3D Spinner */}
                    <div className="relative w-24 h-24 mx-auto mb-6">
                      {/* Outer rotating ring */}
                      <motion.div 
                        className="absolute inset-0 border-4 border-transparent border-t-primary-500 border-r-primary-400 rounded-full"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                      />
                      {/* Middle rotating ring */}
                      <motion.div 
                        className="absolute inset-2 border-4 border-transparent border-t-secondary-500 border-l-secondary-400 rounded-full"
                        animate={{ rotate: -360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      />
                      {/* Inner pulsing circle */}
                      <motion.div 
                        className="absolute inset-6 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-full"
                        animate={{ 
                          scale: [1, 1.2, 1],
                          opacity: [0.5, 0.8, 0.5]
                        }}
                        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                      />
                      {/* Center icon */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Sparkles className="w-8 h-8 text-white" />
                      </div>
                    </div>

                    {/* Animated text */}
                    <motion.div
                      animate={{ opacity: [1, 0.6, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      <p className="text-lg font-semibold text-gray-800 mb-2">
                        Creating Your Floor Plan
                      </p>
                      <p className="text-sm text-gray-600">
                        SDXL Model • GPU Accelerated
                      </p>
                    </motion.div>
                    
                    {/* Animated dots */}
                    <div className="flex justify-center space-x-2 mt-4">
                      {[0, 1, 2].map((i) => (
                        <motion.div
                          key={i}
                          className="w-2 h-2 bg-primary-500 rounded-full"
                          animate={{
                            y: [0, -10, 0],
                            opacity: [0.5, 1, 0.5]
                          }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            delay: i * 0.2
                          }}
                        />
                      ))}
                    </div>

                    {/* Progress bar with gradient */}
                    <div className="mt-6 w-64 mx-auto">
                      <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
                        <motion.div 
                          className="h-full bg-gradient-to-r from-primary-500 via-secondary-500 to-primary-500 bg-[length:200%_100%]"
                          initial={{ width: "0%" }}
                          animate={{ 
                            width: "100%",
                            backgroundPosition: ["0% 0%", "100% 0%"]
                          }}
                          transition={{ 
                            width: { duration: estimatedTime, ease: "easeOut" },
                            backgroundPosition: { duration: 2, repeat: Infinity, ease: "linear" }
                          }}
                        />
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        {modelInfo?.device === 'cuda' && 'Estimated time: ~3-5 seconds ⚡⚡⚡'}
                        {modelInfo?.device === 'mps' && 'Estimated time: ~5-8 seconds ⚡⚡'}
                        {(!modelInfo?.device || modelInfo?.device === 'cpu') && 'Estimated time: ~1-2 minutes ⚡'}
                      </p>
                    </div>

                    {/* Floating particles effect */}
                    <div className="absolute inset-0 pointer-events-none">
                      {[...Array(6)].map((_, i) => (
                        <motion.div
                          key={i}
                          className="absolute w-1 h-1 bg-primary-400 rounded-full"
                          style={{
                            left: `${20 + i * 15}%`,
                            top: '50%'
                          }}
                          animate={{
                            y: [-20, -60, -20],
                            opacity: [0, 1, 0],
                            scale: [0, 1, 0]
                          }}
                          transition={{
                            duration: 3,
                            repeat: Infinity,
                            delay: i * 0.5
                          }}
                        />
                      ))}
                    </div>
                  </motion.div>
                ) : generatedImage ? (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="w-full h-full rounded-lg overflow-hidden relative"
                  >
                    {/* Display the actual generated image */}
                    <img 
                      src={generatedImage} 
                      alt="Generated Floor Plan"
                      className="w-full h-full object-contain bg-white"
                      onError={(e) => {
                        console.error('Image load error:', e);
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                      }}
                    />
                    
                    {/* Fallback display if image fails to load */}
                    <div className="w-full h-full bg-gradient-to-br from-primary-50 to-secondary-50 rounded-lg flex items-center justify-center absolute top-0 left-0" style={{display: 'none'}}>
                      <div className="text-center text-gray-600">
                        <ImageIcon className="w-20 h-20 mx-auto mb-4 text-primary-500" />
                        <p className="font-semibold text-lg">Floor Plan Generated</p>
                        <p className="text-sm text-primary-600 mt-1">Image ready for download</p>
                      </div>
                    </div>
                    
                    {/* Success Animation */}
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: [0, 1.2, 1] }}
                      transition={{ delay: 0.5, duration: 0.5 }}
                      className="absolute top-4 right-4 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center shadow-lg"
                    >
                      <span className="text-white text-sm">✓</span>
                    </motion.div>
                  </motion.div>
                ) : (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center text-gray-400"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.05, 1] }}
                      transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                    >
                      <ImageIcon className="w-20 h-20 mx-auto mb-4" />
                    </motion.div>
                    <p className="text-lg font-medium">Your floor plan will appear here</p>
                    <p className="text-sm mt-2">Enter a description and click generate to start</p>
                  </motion.div>
                )}
              </div>
            </div>



            {/* Tips */}
            <div className="bg-gradient-to-r from-primary-50 to-secondary-50 rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">
                <Wand2 className="w-5 h-5 inline mr-2" />
                Pro Tips
              </h3>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• Be specific about room types and their relationships</li>
                <li>• Mention desired adjacencies (e.g., "kitchen next to dining room")</li>
                <li>• Include approximate sizes or room counts for better results</li>
                <li>• Use architectural terms for more precise layouts</li>
              </ul>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default GeneratorPage;