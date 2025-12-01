import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { History, Download, Trash2, Search, Calendar, Image as ImageIcon, X } from 'lucide-react';
import toast from 'react-hot-toast';

const HistoryPage = () => {
  const [history, setHistory] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [filterStyle, setFilterStyle] = useState('all');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = () => {
    const savedHistory = JSON.parse(localStorage.getItem('floorplanHistory') || '[]');
    setHistory(savedHistory);
  };

  const handleDownload = (item) => {
    try {
      const link = document.createElement('a');
      link.href = item.image;
      link.download = `floorplan-${item.id}.png`;
      link.click();
      toast.success('Floor plan downloaded!');
    } catch (error) {
      toast.error('Failed to download image');
    }
  };

  const handleDelete = (id) => {
    const updatedHistory = history.filter(item => item.id !== id);
    setHistory(updatedHistory);
    localStorage.setItem('floorplanHistory', JSON.stringify(updatedHistory));
    toast.success('Item removed from history');
  };

  const handleClearAll = () => {
    if (window.confirm('Are you sure you want to clear all history? This cannot be undone.')) {
      setHistory([]);
      localStorage.removeItem('floorplanHistory');
      toast.success('History cleared');
    }
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined 
    });
  };

  // Filter history
  const filteredHistory = history.filter(item => {
    const matchesSearch = item.prompt.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStyle = filterStyle === 'all' || item.style === filterStyle;
    return matchesSearch && matchesStyle;
  });

  // Get unique styles
  const styles = ['all', ...new Set(history.map(item => item.style).filter(Boolean))];

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-2">
                <History className="inline-block w-10 h-10 mr-3 text-primary-600" />
                <span className="gradient-text">Generation History</span>
              </h1>
              <p className="text-xl text-gray-600">
                View and manage your generated floor plans
              </p>
            </div>
            {history.length > 0 && (
              <button
                onClick={handleClearAll}
                className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors flex items-center space-x-2"
              >
                <Trash2 className="w-4 h-4" />
                <span>Clear All</span>
              </button>
            )}
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Total Generated</p>
                  <p className="text-2xl font-bold text-gray-900">{history.length}</p>
                </div>
                <ImageIcon className="w-8 h-8 text-primary-600" />
              </div>
            </div>
            <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Unique Styles</p>
                  <p className="text-2xl font-bold text-gray-900">{styles.length - 1}</p>
                </div>
                <Calendar className="w-8 h-8 text-secondary-600" />
              </div>
            </div>
            <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Latest Generation</p>
                  <p className="text-sm font-medium text-gray-900">
                    {history.length > 0 ? formatDate(history[0].timestamp) : 'None'}
                  </p>
                </div>
                <History className="w-8 h-8 text-green-600" />
              </div>
            </div>
          </div>

          {/* Search and Filter */}
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search floor plans..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
            <select
              value={filterStyle}
              onChange={(e) => setFilterStyle(e.target.value)}
              className="px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              {styles.map(style => (
                <option key={style} value={style}>
                  {style === 'all' ? 'All Styles' : style.charAt(0).toUpperCase() + style.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </motion.div>

        {/* History Grid */}
        {filteredHistory.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <ImageIcon className="w-20 h-20 mx-auto mb-4 text-gray-300" />
            <h3 className="text-2xl font-semibold text-gray-600 mb-2">
              {history.length === 0 ? 'No History Yet' : 'No Results Found'}
            </h3>
            <p className="text-gray-500 mb-6">
              {history.length === 0 
                ? 'Generate your first floor plan to see it here'
                : 'Try adjusting your search or filter'}
            </p>
            {history.length === 0 && (
              <a
                href="/"
                className="inline-flex items-center px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                Generate Floor Plan
              </a>
            )}
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredHistory.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-lg transition-shadow"
              >
                {/* Image */}
                <div 
                  className="relative aspect-square bg-gray-100 cursor-pointer group"
                  onClick={() => setSelectedImage(item)}
                >
                  <img
                    src={item.image}
                    alt={item.prompt}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all flex items-center justify-center">
                    <span className="text-white opacity-0 group-hover:opacity-100 transition-opacity font-medium">
                      Click to view
                    </span>
                  </div>
                </div>

                {/* Info */}
                <div className="p-4">
                  <p className="text-sm text-gray-600 mb-2 line-clamp-2">
                    {item.prompt}
                  </p>
                  
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs px-2 py-1 bg-primary-100 text-primary-700 rounded-full">
                      {item.style || 'modern'}
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatDate(item.timestamp)}
                    </span>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleDownload(item)}
                      className="flex-1 px-3 py-2 bg-primary-100 text-primary-700 rounded-lg hover:bg-primary-200 transition-colors flex items-center justify-center space-x-1 text-sm"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download</span>
                    </button>
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="px-3 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}

        {/* Image Modal */}
        <AnimatePresence>
          {selectedImage && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
              onClick={() => setSelectedImage(null)}
            >
              <motion.div
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0.9 }}
                className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-auto"
                onClick={(e) => e.stopPropagation()}
              >
                {/* Modal Header */}
                <div className="sticky top-0 bg-white border-b border-gray-200 p-4 flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Floor Plan Details</h3>
                  <button
                    onClick={() => setSelectedImage(null)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                {/* Modal Content */}
                <div className="p-6">
                  <img
                    src={selectedImage.image}
                    alt={selectedImage.prompt}
                    className="w-full rounded-lg mb-4"
                  />
                  
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm font-semibold text-gray-700">Description</label>
                      <p className="text-gray-600">{selectedImage.prompt}</p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-sm font-semibold text-gray-700">Style</label>
                        <p className="text-gray-600 capitalize">{selectedImage.style || 'modern'}</p>
                      </div>
                      <div>
                        <label className="text-sm font-semibold text-gray-700">Generated</label>
                        <p className="text-gray-600">{formatDate(selectedImage.timestamp)}</p>
                      </div>
                    </div>

                    {selectedImage.metadata && (
                      <div>
                        <label className="text-sm font-semibold text-gray-700">Metadata</label>
                        <div className="text-xs text-gray-600 bg-gray-50 p-3 rounded-lg mt-1">
                          <p>Model: {selectedImage.metadata.model_type || 'SDXL'}</p>
                          <p>Steps: {selectedImage.metadata.steps || 30}</p>
                          <p>Guidance: {selectedImage.metadata.guidance || 7.5}</p>
                        </div>
                      </div>
                    )}

                    <div className="flex gap-2 pt-4">
                      <button
                        onClick={() => handleDownload(selectedImage)}
                        className="flex-1 px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center justify-center space-x-2"
                      >
                        <Download className="w-5 h-5" />
                        <span>Download</span>
                      </button>
                      <button
                        onClick={() => {
                          handleDelete(selectedImage.id);
                          setSelectedImage(null);
                        }}
                        className="px-4 py-3 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors flex items-center justify-center space-x-2"
                      >
                        <Trash2 className="w-5 h-5" />
                        <span>Delete</span>
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default HistoryPage;
