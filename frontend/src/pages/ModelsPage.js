import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, BarChart3, CheckCircle, ArrowRight, Settings, Cpu, Clock, Target } from 'lucide-react';

const ModelsPage = () => {
  const [selectedModel, setSelectedModel] = useState('sdxl');

  const models = [
    {
      id: 'sdxl',
      name: 'FloorMind SDXL',
      description: 'Fine-tuned Stable Diffusion XL model optimized for architectural floor plans',
      type: 'Production Model',
      status: 'Active',
      accuracy: 88.5,
      fidScore: 45.2,
      clipScore: 0.82,
      adjacencyScore: 0.85,
      trainingTime: '8.5 hours',
      parameters: '2.6B',
      features: [
        'SDXL architecture (2.6B parameters)',
        'Fine-tuned on 5,050 floor plans',
        'GPU accelerated (CUDA/MPS)',
        'Ultra-fast generation (3-5s)',
        'High-quality outputs',
        'Spatial understanding'
      ],
      useCases: [
        'Professional floor plans',
        'Rapid prototyping',
        'Multiple variations',
        'Production use',
        'Client presentations'
      ],
      color: 'purple'
    },
    // {
    //   id: 'constraint_aware',
    //   name: 'Constraint-Aware Diffusion',
    //   description: 'Enhanced model with spatial consistency and adjacency constraint loss',
    //   type: 'Advanced Model',
    //   status: 'Recommended',
    //   accuracy: 84.5,
    //   fidScore: 57.4,
    //   clipScore: 0.75,
    //   adjacencyScore: 0.73,
    //   trainingTime: '6.8 hours',
    //   parameters: '865M + Constraints',
    //   features: [
    //     'Spatial constraint enforcement',
    //     'Adjacency relationship modeling',
    //     'Enhanced architectural understanding',
    //     'Multi-loss optimization'
    //   ],
    //   useCases: [
    //     'Professional floor plans',
    //     'Architectural design',
    //     'Real estate visualization',
    //     'Production applications'
    //   ],
    //   color: 'purple'
    // }
  ];

  const architectureDetails = {
    sdxl: {
      components: [
        { name: 'Dual Text Encoders', description: 'CLIP ViT-L and OpenCLIP ViT-bigG for enhanced prompt understanding' },
        { name: 'SDXL U-Net', description: 'Large-scale denoising network (2.6B parameters)' },
        { name: 'VAE Decoder', description: 'High-resolution variational autoencoder' },
        { name: 'DPM++ Scheduler', description: 'Fast multi-step scheduler for efficient generation' }
      ],
      trainingProcess: [
        'Fine-tuned on 5,050 architectural floor plans',
        'Optimized for GPU acceleration (CUDA/MPS)',
        'Mixed precision training (FP16)',
        'Spatial relationship learning'
      ]
    }
  };

  const getColorClasses = (color) => {
    const colors = {
      blue: {
        bg: 'bg-blue-50',
        border: 'border-blue-200',
        text: 'text-blue-700',
        accent: 'bg-blue-600'
      },
      purple: {
        bg: 'bg-purple-50',
        border: 'border-purple-200',
        text: 'text-purple-700',
        accent: 'bg-purple-600'
      }
    };
    return colors[color];
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
            <span className="gradient-text">AI Models</span> Overview
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Explore our advanced diffusion models designed for architectural floor plan generation
          </p>
        </motion.div>

        {/* Model Comparison Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {models.map((model, index) => {
            const colors = getColorClasses(model.color);
            return (
              <motion.div
                key={model.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`bg-white rounded-2xl p-8 shadow-sm border-2 transition-all duration-300 cursor-pointer ${
                  selectedModel === model.id 
                    ? `${colors.border} shadow-lg` 
                    : 'border-gray-100 hover:border-gray-200'
                }`}
                onClick={() => setSelectedModel(model.id)}
              >
                <div className="flex items-start justify-between mb-6">
                  <div className="flex items-center space-x-3">
                    <div className={`w-12 h-12 ${colors.accent} rounded-xl flex items-center justify-center`}>
                      <Brain className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-900">{model.name}</h3>
                      <p className="text-sm text-gray-600">{model.type}</p>
                    </div>
                  </div>
                  <span className={`px-3 py-1 ${colors.bg} ${colors.text} text-xs font-semibold rounded-full`}>
                    {model.status}
                  </span>
                </div>

                <p className="text-gray-600 mb-6">{model.description}</p>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">{model.accuracy}%</div>
                    <div className="text-xs text-gray-600">Accuracy</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">{model.fidScore}</div>
                    <div className="text-xs text-gray-600">FID Score</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">{model.clipScore}</div>
                    <div className="text-xs text-gray-600">CLIP Score</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">{model.adjacencyScore}</div>
                    <div className="text-xs text-gray-600">Adjacency</div>
                  </div>
                </div>

                {/* Key Features */}
                <div className="mb-6">
                  <h4 className="font-semibold text-gray-900 mb-3">Key Features</h4>
                  <div className="space-y-2">
                    {model.features.slice(0, 3).map((feature, idx) => (
                      <div key={idx} className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                        <span className="text-sm text-gray-600">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <button
                  className={`w-full py-3 px-4 ${colors.accent} text-white font-semibold rounded-lg hover:opacity-90 transition-opacity flex items-center justify-center space-x-2`}
                >
                  <span>View Details</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </motion.div>
            );
          })}
        </div>

        {/* Detailed Architecture View */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100"
        >
          <div className="flex items-center space-x-3 mb-8">
            <Settings className="w-6 h-6 text-primary-600" />
            <h2 className="text-2xl font-bold text-gray-900">
              {models.find(m => m.id === selectedModel)?.name} Architecture
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Model Specifications */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                <Cpu className="w-5 h-5 inline mr-2" />
                Model Specifications
              </h3>
              <div className="space-y-4">
                {[
                  { label: 'Parameters', value: models.find(m => m.id === selectedModel)?.parameters, icon: Target },
                  { label: 'Training Time', value: models.find(m => m.id === selectedModel)?.trainingTime, icon: Clock },
                  { label: 'Architecture', value: 'Diffusion Transformer', icon: Brain },
                  { label: 'Input Resolution', value: '512x512', icon: Settings }
                ].map((spec, index) => {
                  const Icon = spec.icon;
                  return (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <Icon className="w-4 h-4 text-gray-600" />
                        <span className="font-medium text-gray-900">{spec.label}</span>
                      </div>
                      <span className="text-gray-600">{spec.value}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Architecture Components */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                <Brain className="w-5 h-5 inline mr-2" />
                Architecture Components
              </h3>
              <div className="space-y-3">
                {architectureDetails[selectedModel]?.components.map((component, index) => (
                  <div key={index} className="p-4 border border-gray-200 rounded-lg">
                    <div className="font-medium text-gray-900 mb-1">{component.name}</div>
                    <div className="text-sm text-gray-600">{component.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Training Process */}
          <div className="mt-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              <BarChart3 className="w-5 h-5 inline mr-2" />
              Training Process
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {architectureDetails[selectedModel]?.trainingProcess.map((step, index) => (
                <div key={index} className="text-center">
                  <div className="w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-sm font-bold">
                    {index + 1}
                  </div>
                  <div className="text-sm font-medium text-gray-900 mb-1">Step {index + 1}</div>
                  <div className="text-xs text-gray-600">{step}</div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Performance Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 bg-gradient-to-r from-primary-50 to-secondary-50 rounded-2xl p-8"
        >
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
            FloorMind SDXL Performance
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { metric: 'Accuracy', value: 88.5, unit: '%', icon: Target, color: 'green' },
              { metric: 'FID Score', value: 45.2, unit: '', icon: BarChart3, color: 'blue' },
              { metric: 'CLIP Score', value: 0.82, unit: '', icon: CheckCircle, color: 'purple' },
              { metric: 'Generation Speed', value: '3-5s', unit: '', icon: Zap, color: 'yellow' }
            ].map((metric, index) => {
              const Icon = metric.icon;
              const colorClasses = {
                green: 'text-green-600 bg-green-50',
                blue: 'text-blue-600 bg-blue-50',
                purple: 'text-purple-600 bg-purple-50',
                yellow: 'text-yellow-600 bg-yellow-50'
              };
              
              return (
                <div key={index} className="bg-white rounded-xl p-6 text-center border border-gray-100 hover:shadow-lg transition-shadow">
                  <div className={`w-12 h-12 rounded-full ${colorClasses[metric.color]} flex items-center justify-center mx-auto mb-4`}>
                    <Icon className="w-6 h-6" />
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-2">{metric.metric}</h3>
                  <div className="text-3xl font-bold text-gray-900 mb-1">
                    {metric.value}{metric.unit}
                  </div>
                  <div className="text-xs text-gray-500">SDXL Model</div>
                </div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default ModelsPage;