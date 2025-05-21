import React, { Component } from 'react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false,
      error: null,
      errorInfo: null 
    };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to an error reporting service
    console.error('Error caught by ErrorBoundary:', error, errorInfo);
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  render() {
    if (this.state.hasError) {
      // Fallback UI when an error occurs
      return (
        <div className="p-6 bg-[#2a2d37] text-white rounded-lg shadow-md flex flex-col items-center justify-center min-h-[200px]">
          <h2 className="text-2xl font-bold text-red-400 mb-4">Something went wrong</h2>
          <p className="text-[#d1d5db] mb-4">
            There was an error in the application. This could be due to a network issue or a problem with the backend service.
          </p>
          <details className="bg-[#1e2129] p-4 rounded-lg w-full mb-4 custom-scrollbar overflow-auto max-h-[200px]">
            <summary className="text-[#3b82f6] cursor-pointer mb-2">Error Details</summary>
            <p className="text-[#d1d5db] whitespace-pre-wrap">
              {this.state.error && this.state.error.toString()}
            </p>
            <p className="text-[#8e8ea0] mt-2 whitespace-pre-wrap">
              {this.state.errorInfo && this.state.errorInfo.componentStack}
            </p>
          </details>
          <button
            onClick={() => {
              // Reset the error state and refresh the page
              this.setState({ hasError: false, error: null, errorInfo: null });
              window.location.reload();
            }}
            className="p-2 bg-[#3b82f6] text-white rounded-md hover:bg-[#2563eb] transition-colors"
          >
            Try Again
          </button>
        </div>
      );
    }

    // Render children if no error
    return this.props.children;
  }
}

export default ErrorBoundary; 