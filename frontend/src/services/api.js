import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

// Add a response interceptor to handle errors gracefully
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const apiService = {
  getDashboardStats: () => api.get('/dashboard/stats'),
  
  getCandidates: (search = '') => api.get('/candidates', { params: { search } }),
  getCandidate: (id) => api.get(`/candidates/${id}`),
  getCandidateDocuments: (candidateId) => api.get(`/candidates/${candidateId}/documents`),
  
  getDocuments: (status = '') => api.get('/documents', { params: { status } }),
  getDocument: (id) => api.get(`/documents/${id}`),
  getDocumentStatus: (id) => api.get(`/documents/${id}/status`),
  getDocumentOcr: (id) => api.get(`/documents/${id}/ocr`),
  getDocumentClassification: (id) => api.get(`/documents/${id}/classification`),
  
  classifyDocument: (id) => api.post(`/documents/${id}/classify`),
  
  uploadDocument: (file, candidateId) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('candidate_id', candidateId);
    return api.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  getDocumentImageUrl: (id) => `/api/documents/${id}/image`,
  getProcessedImageUrl: (id) => `/api/documents/${id}/processed-image`,
  
  getReviewDocuments: () => api.get('/documents/review'),
  syncStorage: () => api.post('/documents/sync'),
  resetScrutiny: () => api.post('/documents/reset'),
};
