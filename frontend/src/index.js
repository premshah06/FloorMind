import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import AppWithRouter from './AppWithRouter';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <AppWithRouter />
  </React.StrictMode>
);