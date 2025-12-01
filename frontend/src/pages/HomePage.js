import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Sparkles, Cpu, TrendingUp, Rocket, CheckCircle2, PlayCircle } from 'lucide-react';
import { motion } from 'framer-motion';

const HomePage = () => {
  const features = [
    {
      icon: Cpu,
      title: 'AI-Powered Generation',
      description: 'Advanced diffusion models fine-tuned on architectural datasets for precise floor plan generation.',
      color: 'from-teal-500 to-cyan-500'
    },
    // {
    //   icon: Sparkles,
    //   title: 'Constraint-Aware Design',
    //   description: 'Ensures spatial consistency and adjacency relationships for realistic architectural layouts.',
    //   color: 'from-purple-500 to-fuchsia-500'
    // },
    {
      icon: TrendingUp,
      title: 'Performance Metrics',
      description: 'Comprehensive evaluation using FID, CLIP-Score, and adjacency consistency metrics.',
      color: 'from-blue-500 to-indigo-500'
    },
    {
      icon: Rocket,
      title: 'Real-time Generation',
      description: 'Fast inference with optimized models for quick floor plan generation from text prompts.',
      color: 'from-orange-500 to-pink-500'
    }
  ];

  // Removed stats section - keeping UI clean and focused

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary-50 via-white to-secondary-50 py-20 sm:py-32">
        {/* Animated Background Elements */}
        <div className="absolute inset-0">
          <div className="absolute top-20 left-10 w-72 h-72 bg-primary-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
          <div className="absolute top-40 right-10 w-72 h-72 bg-secondary-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse" style={{animationDelay: '2s'}}></div>
          <div className="absolute -bottom-8 left-20 w-72 h-72 bg-accent-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse" style={{animationDelay: '4s'}}></div>
        </div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={containerVariants}
            className="text-center"
          >
            <motion.h1 
              variants={itemVariants}
              className="text-4xl sm:text-6xl lg:text-7xl font-bold text-gray-900 mb-6"
            >
              <motion.span 
                className="gradient-text inline-block"
                animate={{ 
                  backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
                }}
                transition={{ 
                  duration: 5, 
                  repeat: Infinity, 
                  ease: "linear" 
                }}
              >
                FloorMind
              </motion.span>
              <br />
              <span className="text-gray-700">AI Floor Plans</span>
            </motion.h1>
            
            <motion.p 
              variants={itemVariants}
              className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed"
            >
              Transform natural language descriptions into detailed architectural floor plans 
              using advanced diffusion models with spatial constraints and adjacency awareness.
            </motion.p>
            
            <motion.div 
              variants={itemVariants}
              className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            >
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link
                  to="/generate"
                  className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-primary-600 to-secondary-600 text-white font-semibold rounded-xl hover:from-primary-700 hover:to-secondary-700 transition-all duration-200 shadow-lg hover:shadow-xl"
                >
                  <Sparkles className="w-5 h-5 mr-2" />
                  Try Generator
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Link>
              </motion.div>
              
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link
                  to="/models"
                  className="inline-flex items-center px-8 py-4 bg-white text-gray-700 font-semibold rounded-xl border-2 border-gray-200 hover:border-primary-300 hover:text-primary-700 transition-all duration-200 shadow-sm hover:shadow-md"
                >
                  <PlayCircle className="w-5 h-5 mr-2" />
                  View Models
                </Link>
              </motion.div>
            </motion.div>

            {/* Live Demo Preview */}
            <motion.div
              variants={itemVariants}
              className="mt-16"
            >
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-lg border border-gray-200 max-w-4xl mx-auto">
                <div className="flex items-center justify-center space-x-4 mb-6">
                  <div className="flex space-x-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  </div>
                  <span className="text-sm font-medium text-gray-600">FloorMind Generator</span>
                </div>
                <div className="bg-gray-50 rounded-lg p-4 text-left">
                  <div className="text-sm text-gray-500 mb-2">Input:</div>
                  <div className="font-mono text-sm text-gray-800 mb-4">
                    "3-bedroom apartment with open kitchen and living room"
                  </div>
                  <div className="text-sm text-gray-500 mb-2">Output:</div>
                  <div className="bg-gradient-to-r from-primary-100 to-secondary-100 rounded-lg p-6 text-center">
                    <div className="w-16 h-16 mx-auto mb-3 bg-white rounded-lg flex items-center justify-center">
                      <svg className="w-10 h-10 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                      </svg>
                    </div>
                    <div className="text-sm text-gray-600 font-medium">Generated Floor Plan</div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>



      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={containerVariants}
            className="text-center mb-16"
          >
            <motion.h2 
              variants={itemVariants}
              className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4"
            >
              Powerful AI Features
            </motion.h2>
            <motion.p 
              variants={itemVariants}
              className="text-xl text-gray-600 max-w-3xl mx-auto"
            >
              Built with cutting-edge machine learning techniques and architectural expertise
            </motion.p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={containerVariants}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8"
          >
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  variants={itemVariants}
                  className="bg-white rounded-2xl p-8 shadow-sm hover:shadow-lg transition-all duration-300 card-hover"
                >
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${feature.color} flex items-center justify-center mb-6`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {feature.description}
                  </p>
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      </section>



      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-primary-600 to-secondary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={containerVariants}
          >
            <motion.h2 
              variants={itemVariants}
              className="text-3xl lg:text-4xl font-bold text-white mb-4"
            >
              Ready to Generate Floor Plans?
            </motion.h2>
            <motion.p 
              variants={itemVariants}
              className="text-xl text-primary-100 mb-8 max-w-2xl mx-auto"
            >
              Experience the power of AI-driven architectural design with FloorMind's 
              advanced text-to-floorplan generation.
            </motion.p>
            <motion.div variants={itemVariants}>
              <Link
                to="/generate"
                className="inline-flex items-center px-8 py-4 bg-white text-primary-600 font-semibold rounded-xl hover:bg-gray-50 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-1"
              >
                <Sparkles className="w-5 h-5 mr-2" />
                Start Generating
                <ArrowRight className="w-5 h-5 ml-2" />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;