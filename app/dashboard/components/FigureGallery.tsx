'use client';

import { useState } from 'react';
import Image from 'next/image';

interface FigureGalleryProps {
  figures: string[];
  basePath?: string;
  phasePath?: string;
}

export default function FigureGallery({ figures, basePath = 'figures', phasePath }: FigureGalleryProps) {
  const [selectedFigure, setSelectedFigure] = useState<string | null>(null);
  const [imageErrors, setImageErrors] = useState<Set<string>>(new Set());

  const handleImageError = (figure: string) => {
    setImageErrors(prev => new Set(prev).add(figure));
  };

  const getImageUrl = (figure: string) => {
    // If phasePath is provided, use it; otherwise default to phase1_eda
    const phase = phasePath || 'phase1_eda';
    return `/${phase}/${basePath}/${figure}`;
  };

  const getFigureTitle = (filename: string) => {
    return filename
      .replace('.png', '')
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (figures.length === 0) {
    return (
      <div className="text-gray-500 italic text-sm">
        No figures generated
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {figures.map((figure, index) => (
          <div
            key={index}
            className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow cursor-pointer bg-white"
            onClick={() => setSelectedFigure(figure)}
          >
            <div className="aspect-video relative bg-gray-50">
              {!imageErrors.has(figure) ? (
                <img
                  src={getImageUrl(figure)}
                  alt={getFigureTitle(figure)}
                  className="w-full h-full object-contain"
                  onError={() => handleImageError(figure)}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-400">
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
              )}
            </div>
            <div className="p-2 bg-white border-t border-gray-100">
              <p className="text-xs text-gray-700 font-medium truncate" title={getFigureTitle(figure)}>
                {getFigureTitle(figure)}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for full-size view */}
      {selectedFigure && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedFigure(null)}
        >
          <div className="relative max-w-6xl max-h-[90vh] bg-white rounded-lg overflow-hidden">
            <button
              onClick={() => setSelectedFigure(null)}
              className="absolute top-2 right-2 bg-white rounded-full p-2 shadow-lg hover:bg-gray-100 z-10"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <div className="p-4">
              <h3 className="text-xl font-semibold mb-4 text-gray-900">
                {getFigureTitle(selectedFigure)}
              </h3>
              <div className="overflow-auto max-h-[calc(90vh-8rem)]">
                <img
                  src={getImageUrl(selectedFigure)}
                  alt={getFigureTitle(selectedFigure)}
                  className="w-full h-auto"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
