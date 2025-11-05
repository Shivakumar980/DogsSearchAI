import React, { useState, useEffect } from 'react';
import './DogsUI.css';

const DogsUI = () => {
  const [dogs, setDogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [progressMessage, setProgressMessage] = useState('');
  const [hasSearched, setHasSearched] = useState(false);
  const [currentCarouselIndex, setCurrentCarouselIndex] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const resultsPerPage = 12;

  // WebSocket endpoint
  const WS_URL = 'ws://localhost:8000/ws/search';

  useEffect(() => {
    // Connect to WebSocket
    const websocket = new WebSocket(WS_URL);

    websocket.onopen = () => {
      console.log('WebSocket Connected');
      setIsConnected(true);
      setError(null);
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received data:', data);
        
        // Handle different message types from backend
        if (data.type === 'results' && data.data && data.data.results) {
          // Final results
          setDogs(data.data.results);
          setLoading(false);
          setProgressMessage('');
          setHasSearched(true);
        } else if (data.type === 'progress') {
          // Progress update
          console.log('Progress:', data.message);
          setProgressMessage(data.message || '');
          // Keep loading state
        } else if (data.type === 'search_started') {
          // Search started
          console.log('Search started for:', data.query);
          setDogs([]); // Clear previous results
          setLoading(true);
          setProgressMessage('Starting search...');
          setError(null);
          setHasSearched(true);
        } else if (data.type === 'error') {
          // Error message
          setError(data.message || 'Search failed');
          setLoading(false);
        } else if (data.type === 'pong') {
          // Health check response
          console.log('Pong received');
        } else if (data.results) {
          // Fallback for direct results
          setDogs(data.results);
          setLoading(false);
        } else if (Array.isArray(data)) {
          // Fallback for array results
          setDogs(data);
          setLoading(false);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
        setError('Failed to parse dog data');
        setLoading(false);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection error. Please make sure the server is running.');
      setIsConnected(false);
      setLoading(false);
    };

    websocket.onclose = () => {
      console.log('WebSocket Disconnected');
      setIsConnected(false);
    };

    setWs(websocket);

    // Cleanup on unmount
    return () => {
      websocket.close();
    };
  }, []);

  const searchDogs = (query = '') => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError('WebSocket is not connected. Please refresh the page.');
      return;
    }

    setLoading(true);
    setError(null);

    // Send search query through WebSocket (backend expects type: "search")
    const searchQuery = {
      type: 'search',
      query: query || searchTerm,
      top_k: 24,  // Total results to fetch (12 per page × 2 pages)
      rerank_top_n: 72  // Increased for better quality with 24 results
    };
    
    // Reset to first page on new search
    setCurrentPage(1);

    console.log('Sending search query:', searchQuery);
    ws.send(JSON.stringify(searchQuery));
  };

  const handleSearch = () => {
    searchDogs(searchTerm);
  };

  const handleRefresh = () => {
    // Send empty query to get all dogs or initial results
    searchDogs('');
  };

  // Carousel images - all images from public/images folder
  const carouselImages = [
    '/images/dog (1).jpeg',
    '/images/_ (5).jpeg',
    '/images/_ (6).jpeg',
    '/images/_ (7).jpeg',
    '/images/HH Bath View Apartments Pet Policy _ Breed Restrictions & Registration.jpeg',
    '/images/Smiley woman and dog with tablet medium shot _ Free Photo.jpeg'
  ];

  // Auto-slide carousel
  useEffect(() => {
    if (!hasSearched && carouselImages.length > 0) {
      const interval = setInterval(() => {
        setCurrentCarouselIndex((prev) => (prev + 1) % carouselImages.length);
      }, 4000); // Change slide every 4 seconds
      return () => clearInterval(interval);
    }
  }, [hasSearched, carouselImages.length]);

  return (
    <div className="spectrum-app">
      {/* Header */}
      <header className="spectrum-header">
        <div className="header-content">
          <div className="header-main">
            <h1 className="spectrum-heading">
              <span className="icon">🐕</span>
              Your Next Best Friend
            </h1>
            <p className="header-tagline">Discover the perfect dog breed for your life with AI powered search</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="spectrum-main">
        <div className="container">
          {/* Search Bar */}
          <div className="search-section">
            <div className="spectrum-search">
              <input
                type="text"
                className="spectrum-textfield"
                placeholder="Search by name, group, or temperament..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleSearch();
                  }
                }}
              />
              <svg className="search-icon" width="20" height="20" viewBox="0 0 20 20">
                <path d="M8.5 3C11.5376 3 14 5.46243 14 8.5C14 9.83879 13.5217 11.0659 12.7266 12.0196L16.8536 16.1464C17.0488 16.3417 17.0488 16.6583 16.8536 16.8536C16.68 17.0271 16.4106 17.0464 16.2157 16.9114L16.1464 16.8536L12.0196 12.7266C11.0659 13.5217 9.83879 14 8.5 14C5.46243 14 3 11.5376 3 8.5C3 5.46243 5.46243 3 8.5 3Z" fill="currentColor"/>
              </svg>
            </div>
            <button className="spectrum-button spectrum-button--primary" onClick={handleSearch}>
              Search
            </button>
            <div className="connection-status-wrapper">
              {isConnected ? (
                <div className="connection-status connected">
                  <svg className="status-icon" width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <circle cx="6" cy="6" r="5" fill="#4ADE80"/>
                    <path d="M4 6L5.5 7.5L8 5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span>Connected</span>
                </div>
              ) : (
                <div className="connection-status disconnected">
                  <svg className="status-icon" width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <circle cx="6" cy="6" r="5" fill="#F87171"/>
                    <path d="M4 4L8 8M8 4L4 8" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  <span>Disconnected</span>
                </div>
              )}
            </div>
          </div>

          {/* Carousel - Show when no search has been performed */}
          {!hasSearched && carouselImages.length > 0 && (
            <div className="carousel-container">
              <div className="carousel-wrapper">
                {carouselImages.map((imageUrl, index) => (
                  <div
                    key={index}
                    className={`carousel-slide ${index === currentCarouselIndex ? 'active' : ''}`}
                  >
                    <img src={imageUrl} alt={`Dog breed ${index + 1}`} className="carousel-image" />
                  </div>
                ))}
              </div>
              {/* Carousel Indicators */}
              <div className="carousel-indicators">
                {carouselImages.map((_, index) => (
                  <button
                    key={index}
                    className={`carousel-indicator ${index === currentCarouselIndex ? 'active' : ''}`}
                    onClick={() => setCurrentCarouselIndex(index)}
                    aria-label={`Go to slide ${index + 1}`}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="spectrum-loading">
              <div className="spinner"></div>
              <p>{progressMessage || 'Loading dogs...'}</p>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="spectrum-alert spectrum-alert--error">
              <svg className="alert-icon" width="20" height="20" viewBox="0 0 20 20">
                <path d="M10 2C14.4183 2 18 5.58172 18 10C18 14.4183 14.4183 18 10 18C5.58172 18 2 14.4183 2 10C2 5.58172 5.58172 2 10 2ZM10 11C9.44772 11 9 11.4477 9 12C9 12.5523 9.44772 13 10 13C10.5523 13 11 12.5523 11 12C11 11.4477 10.5523 11 10 11ZM10 6C9.44772 6 9 6.44772 9 7V9C9 9.55228 9.44772 10 10 10C10.5523 10 11 9.55228 11 9V7C11 6.44772 10.5523 6 10 6Z" fill="currentColor"/>
              </svg>
              {error}
            </div>
          )}

          {/* Dogs Grid */}
          {!loading && !error && (
            <div className="dogs-grid-wrapper">
              {dogs.length > 0 ? (
                <>
                  {/* Paginated Results */}
                  <div className="dogs-grid">
                    {dogs
                      .slice((currentPage - 1) * resultsPerPage, currentPage * resultsPerPage)
                      .map((dog, index) => (
                  <div key={dog.name || index} className="spectrum-card">
                    <div className="card-image">
                      {dog.image_url ? (
                        <img src={dog.image_url} alt={dog.name} />
                      ) : (
                        <div className="placeholder-image">🐕</div>
                      )}
                    </div>
                    <div className="card-content">
                      <div className="card-header-row">
                        <h3 className="card-title">{dog.name}</h3>
                        {dog.match_category && (
                          <div 
                            className="match-category-badge"
                            style={{
                              backgroundColor: dog.match_category.color,
                              color: 'white'
                            }}
                            title={dog.match_category.description}
                          >
                            <span className="category-icon">{dog.match_category.icon || ''}</span>
                            <span className="category-label">{dog.match_category.label}</span>
                          </div>
                        )}
                      </div>
                      
                      <div className="card-info-grid">
                        <div className="info-item">
                          <span className="info-label">Size:</span>
                          <span className="info-value">{dog.size || 'Unknown'}</span>
                        </div>
                        <div className="info-item">
                          <span className="info-label">Weight:</span>
                          <span className="info-value">{dog.weight || 'N/A'}</span>
                        </div>
                        <div className="info-item">
                          <span className="info-label">Height:</span>
                          <span className="info-value">{dog.height || 'N/A'}</span>
                        </div>
                      </div>

                      <div className="card-section">
                        <div className="info-item">
                          <span className="info-label">Bred for:</span>
                          <span className="info-value">{dog.bred_for || 'N/A'}</span>
                        </div>
                        <div className="info-item">
                          <span className="info-label">Lifespan:</span>
                          <span className="info-value">{dog.lifespan || 'N/A'}</span>
                        </div>
                      </div>

                      <div className="card-temperament">
                        <span className="info-label">Temperament:</span>
                        <p className="temperament-text">{dog.temperament || 'N/A'}</p>
                      </div>
                    </div>
                  </div>
                      ))}
                  </div>
                  
                  {/* Pagination Controls */}
                  {dogs.length > resultsPerPage && (
                    <div className="pagination">
                      <button
                        className="pagination-button"
                        onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                        disabled={currentPage === 1}
                      >
                        Previous
                      </button>
                      <span className="pagination-info">
                        Page {currentPage} of {Math.ceil(dogs.length / resultsPerPage)}
                      </span>
                      <button
                        className="pagination-button"
                        onClick={() => setCurrentPage(prev => Math.min(Math.ceil(dogs.length / resultsPerPage), prev + 1))}
                        disabled={currentPage >= Math.ceil(dogs.length / resultsPerPage)}
                      >
                        Next
                      </button>
                    </div>
                  )}
                </>
              ) : hasSearched ? (
                <div className="spectrum-empty-state">
                  <p>No dogs found matching your search.</p>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default DogsUI;