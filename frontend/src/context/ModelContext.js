import React, { createContext, useContext, useState, useEffect } from 'react';

const ModelContext = createContext();

export const useModel = () => {
  const context = useContext(ModelContext);
  if (!context) {
    throw new Error('useModel must be used within ModelProvider');
  }
  return context;
};

export const ModelProvider = ({ children }) => {
  // Load from localStorage on mount
  const loadFromStorage = () => {
    try {
      const saved = localStorage.getItem('floorMindModelState');
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (error) {
      console.error('Error loading model state:', error);
    }
    return { modelLoaded: false, modelInfo: null };
  };

  const initialState = loadFromStorage();
  
  const [modelLoaded, setModelLoaded] = useState(initialState.modelLoaded);
  const [modelInfo, setModelInfo] = useState(initialState.modelInfo);

  // Save to localStorage whenever state changes
  useEffect(() => {
    const stateToSave = { modelLoaded, modelInfo };
    localStorage.setItem('floorMindModelState', JSON.stringify(stateToSave));
  }, [modelLoaded, modelInfo]);

  const updateModelState = (loaded, info) => {
    setModelLoaded(loaded);
    setModelInfo(info);
  };

  const value = {
    modelLoaded,
    modelInfo,
    setModelLoaded,
    setModelInfo,
    updateModelState
  };

  return (
    <ModelContext.Provider value={value}>
      {children}
    </ModelContext.Provider>
  );
};
