import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, Target, Users, Award, Code, Lightbulb, Rocket, ArrowRight } from 'lucide-react';

const AboutPage = () => {
  const features = [
    {
      icon: Brain,
      title: 'Advanced AI Models',
      description: 'State-of-the-art diffusion models fine-tuned specifically for architectural floor plan generation with spatial constraint awareness.',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Zap,
      title: 'Real-time Generation',
      description: 'Fast inference pipeline optimized for quick floor plan generation from natural language descriptions in under 3 seconds.',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: Target,
      title: 'High Accuracy',
      description: '84.5% generation accuracy with comprehensive evaluation metrics including FID, CLIP-Score, and adjacency consistency.',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: Code,
      title: 'Open Architecture',
      description: 'Modular, extensible codebase built with modern technologies and best practices for easy integration and customization.',
      color: 'from-orange-500 to-red-500'
    }
  ];

  const achievements = [
    {
      category: 'AI Innovation',
      title: 'Advanced Diffusion Models',
      description: 'Developed state-of-the-art diffusion models fine-tuned for architectural floor plan generation.',
      icon: Brain,
      color: 'from-blue-500 to-cyan-500',
      metrics: ['Stable Diffusion 2.1', 'LoRA Fine-tuning', 'CubiCasa5K Dataset']
    },
    {
      category: 'Technical Excellence',
      title: 'Production-Ready Architecture',
      description: 'Built scalable, modular system with comprehensive API and responsive frontend.',
      icon: Code,
      color: 'from-purple-500 to-pink-500',
      metrics: ['React + Flask', 'RESTful API', 'Responsive Design']
    },
    {
      category: 'Performance',
      title: 'Optimized Generation',
      description: 'Fast inference pipeline with memory-efficient attention and gradient checkpointing.',
      icon: Target,
      color: 'from-green-500 to-emerald-500',
      metrics: ['GPU Accelerated', 'CPU Compatible', 'Real-time Processing']
    },
    {
      category: 'User Experience',
      title: 'Intuitive Interface',
      description: 'Designed user-friendly interface for seamless text-to-floorplan generation.',
      icon: Lightbulb,
      color: 'from-orange-500 to-red-500',
      metrics: ['Natural Language', 'One-Click Generation', 'Instant Download']
    }
  ];

  const team = [
    {
      name: 'AI Research Team',
      role: 'Model Development',
      description: 'Specialized in diffusion models, computer vision, and architectural AI applications.',
      icon: Brain
    },
    {
      name: 'Engineering Team',
      role: 'Platform Development',
      description: 'Full-stack developers focused on scalable architecture and user experience.',
      icon: Code
    },
    {
      name: 'Architecture Experts',
      role: 'Domain Knowledge',
      description: 'Professional architects providing domain expertise and validation.',
      icon: Target
    },
    {
      name: 'Product Team',
      role: 'Strategy & Design',
      description: 'Product managers and designers ensuring user-centric development.',
      icon: Lightbulb
    }
  ];

  // Removed stats - keeping page clean and focused on features

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
            About <span className="gradient-text">FloorMind</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
            FloorMind is an innovative AI-powered platform that transforms natural language descriptions 
            into detailed architectural floor plans using advanced diffusion models with spatial constraint awareness.
          </p>
        </motion.div>

        {/* Mission Statement */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-3xl p-12 text-white text-center mb-16"
        >
          <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
          <p className="text-xl leading-relaxed max-w-4xl mx-auto">
            To democratize architectural design by making professional-quality floor plan generation 
            accessible to everyone through cutting-edge AI technology, enabling architects, designers, 
            and homeowners to bring their spatial visions to life effortlessly.
          </p>
        </motion.div>

        {/* Key Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100 hover:shadow-lg transition-all duration-300"
                >
                  <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${feature.color} flex items-center justify-center mb-6`}>
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                </motion.div>
              );
            })}
          </div>
        </motion.div>



        {/* Key Achievements */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Key Achievements</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {achievements.map((achievement, index) => {
              const Icon = achievement.icon;
              return (
                <motion.div
                  key={achievement.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100 hover:shadow-lg transition-all duration-300 group"
                >
                  <div className="flex items-start space-x-4 mb-6">
                    <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${achievement.color} flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
                      <Icon className="w-8 h-8 text-white" />
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-primary-600 mb-1">{achievement.category}</div>
                      <h3 className="text-xl font-bold text-gray-900 mb-2">{achievement.title}</h3>
                    </div>
                  </div>
                  
                  <p className="text-gray-600 leading-relaxed mb-6">{achievement.description}</p>
                  
                  <div className="flex flex-wrap gap-2">
                    {achievement.metrics.map((metric, idx) => (
                      <span key={idx} className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm font-medium">
                        {metric}
                      </span>
                    ))}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Team */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Our Team</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {team.map((member, index) => {
              const Icon = member.icon;
              return (
                <motion.div
                  key={member.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 text-center hover:shadow-lg transition-all duration-300"
                >
                  <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 mb-2">{member.name}</h3>
                  <div className="text-primary-600 font-medium mb-3">{member.role}</div>
                  <p className="text-sm text-gray-600">{member.description}</p>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Technology Stack */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Technology Stack</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: 'AI & Machine Learning',
                icon: Brain,
                color: 'from-blue-500 to-cyan-500',
                technologies: [
                  { name: 'PyTorch & Diffusers', desc: 'Deep learning framework' },
                  { name: 'Stable Diffusion 2.1', desc: 'Foundation model' },
                  { name: 'CLIP & Transformers', desc: 'Text understanding' },
                  { name: 'Custom Constraints', desc: 'Spatial awareness' }
                ]
              },
              {
                title: 'Backend & API',
                icon: Code,
                color: 'from-purple-500 to-pink-500',
                technologies: [
                  { name: 'Flask & Python', desc: 'Web framework' },
                  { name: 'RESTful API', desc: 'Clean architecture' },
                  { name: 'Model Serving', desc: 'Inference pipeline' },
                  { name: 'Monitoring', desc: 'Performance tracking' }
                ]
              },
              {
                title: 'Frontend & UI',
                icon: Target,
                color: 'from-green-500 to-emerald-500',
                technologies: [
                  { name: 'React 18', desc: 'Modern UI framework' },
                  { name: 'Tailwind CSS', desc: 'Utility-first styling' },
                  { name: 'Framer Motion', desc: 'Smooth animations' },
                  { name: 'Responsive Design', desc: 'Mobile-first approach' }
                ]
              }
            ].map((stack, index) => {
              const Icon = stack.icon;
              return (
                <motion.div
                  key={stack.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100 hover:shadow-lg transition-all duration-300"
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${stack.color} flex items-center justify-center`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900">{stack.title}</h3>
                  </div>
                  <div className="space-y-4">
                    {stack.technologies.map((tech, idx) => (
                      <div key={idx} className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-primary-500 mt-2 flex-shrink-0"></div>
                        <div>
                          <div className="font-medium text-gray-900">{tech.name}</div>
                          <div className="text-sm text-gray-600">{tech.desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Future Vision */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <div className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-3xl p-12 text-white relative overflow-hidden">
            <div className="absolute inset-0 bg-black/10"></div>
            <div className="relative z-10">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="w-20 h-20 mx-auto mb-8"
              >
                <Rocket className="w-20 h-20 text-white" />
              </motion.div>
              <h2 className="text-4xl font-bold mb-6">The Future of Architectural AI</h2>
              <p className="text-xl text-primary-100 max-w-4xl mx-auto leading-relaxed mb-12">
                FloorMind represents just the beginning of AI-driven architectural design. We're building 
                the foundation for a future where anyone can create professional-quality architectural 
                designs through natural language, democratizing design and unleashing human creativity.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                {[
                  { icon: Lightbulb, title: '3D Generation', desc: 'Full 3D architectural models' },
                  { icon: Target, title: 'VR Integration', desc: 'Immersive design experience' },
                  { icon: Users, title: 'Collaboration', desc: 'Real-time team design' },
                  { icon: Zap, title: 'Mobile Apps', desc: 'Design on the go' }
                ].map((feature, index) => {
                  const Icon = feature.icon;
                  return (
                    <motion.div
                      key={feature.title}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 + index * 0.1 }}
                      className="bg-white/10 backdrop-blur-sm rounded-xl p-6 hover:bg-white/20 transition-all duration-300"
                    >
                      <div className="w-12 h-12 mx-auto mb-3 bg-white/20 rounded-lg flex items-center justify-center">
                        <Icon className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="font-semibold mb-2">{feature.title}</h3>
                      <p className="text-sm text-primary-100">{feature.desc}</p>
                    </motion.div>
                  );
                })}
              </div>
              
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1 }}
                className="inline-flex items-center space-x-2 bg-white text-primary-600 px-8 py-4 rounded-xl font-semibold hover:bg-primary-50 transition-colors cursor-pointer"
              >
                <span>Join the Revolution</span>
                <ArrowRight className="w-5 h-5" />
              </motion.div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default AboutPage;