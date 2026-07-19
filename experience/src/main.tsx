import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App.js';
import { Explain } from './Explain.js';
import './index.css';

const page = window.location.pathname.replace(/\/+$/, '') || '/';
const Root = page === '/how-it-works' ? Explain : App;

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
