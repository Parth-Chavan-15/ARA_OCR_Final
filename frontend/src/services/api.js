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
  getModelInfo: () => api.get('/model-info'),
};

export const queueService = {
  getQueue: () => {
    try {
      return JSON.parse(localStorage.getItem('ara_scrutiny_queue') || '[]');
    } catch {
      return [];
    }
  },
  isInQueue: (docId) => {
    const q = queueService.getQueue();
    return q.includes(docId);
  },
  addToQueue: (docId) => {
    const q = queueService.getQueue();
    if (!q.includes(docId)) {
      q.push(docId);
      localStorage.setItem('ara_scrutiny_queue', JSON.stringify(q));
      window.dispatchEvent(new Event('ara_queue_updated'));
    }
    return q;
  },
  addAllToQueue: (docIds) => {
    const q = queueService.getQueue();
    docIds.forEach((id) => {
      if (!q.includes(id)) {
        q.push(id);
      }
    });
    localStorage.setItem('ara_scrutiny_queue', JSON.stringify(q));
    window.dispatchEvent(new Event('ara_queue_updated'));
    return q;
  },
  removeFromQueue: (docId) => {
    let q = queueService.getQueue();
    q = q.filter((id) => id !== docId);
    localStorage.setItem('ara_scrutiny_queue', JSON.stringify(q));
    window.dispatchEvent(new Event('ara_queue_updated'));
    return q;
  },
  clearQueue: () => {
    localStorage.removeItem('ara_scrutiny_queue');
    window.dispatchEvent(new Event('ara_queue_updated'));
    return [];
  },
};

export const createWebSocketClient = (onMessage) => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname || 'localhost';
  const port = '8000';
  const wsUrl = `${protocol}//${host}:${port}/ws/scrutiny`;

  let ws = null;
  let reconnectTimer = null;
  let pingTimer = null;
  let isClosedExplicitly = false;

  const connect = () => {
    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        // Heartbeat ping every 25s
        pingTimer = setInterval(() => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
          }
        }, 25000);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.event !== 'pong' && onMessage) {
            onMessage(data);
          }
        } catch (e) {
          console.debug('WS parse error:', e);
        }
      };

      ws.onclose = () => {
        if (pingTimer) clearInterval(pingTimer);
        if (!isClosedExplicitly) {
          reconnectTimer = setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => {
        if (ws) ws.close();
      };
    } catch (err) {
      if (!isClosedExplicitly) {
        reconnectTimer = setTimeout(connect, 3000);
      }
    }
  };

  connect();

  return () => {
    isClosedExplicitly = true;
    if (pingTimer) clearInterval(pingTimer);
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (ws) ws.close();
  };
};
