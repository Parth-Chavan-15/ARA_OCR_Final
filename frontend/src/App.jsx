import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import CandidateDocuments from './pages/CandidateDocuments';
import IncomingDocuments from './pages/IncomingDocuments';
import ClassificationResult from './pages/ClassificationResult';
import ReviewQueue from './pages/ReviewQueue';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="candidates" element={<CandidateDocuments />} />
          <Route path="incoming" element={<IncomingDocuments />} />
          <Route path="result/:id" element={<ClassificationResult />} />
          <Route path="review" element={<ReviewQueue />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
