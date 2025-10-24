'use client';

import { useState } from 'react';
import ValidationSidebar from './components/ValidationSidebar';
import ConsoleOutput from './components/ConsoleOutput';
import ValidationResults from './components/ValidationResults';
import { AnalysisResponse, ValidationResponse } from '@/lib/types';

interface SubphaseResults {
  analysis?: AnalysisResponse;
  validation?: ValidationResponse;
  logs: string[];
}

export default function DashboardPage() {
  const [subphaseResults, setSubphaseResults] = useState<Record<string, SubphaseResults>>({});
  const [activeSubphase, setActiveSubphase] = useState<string>('phase1_1');
  const [activeTab, setActiveTab] = useState<'console' | 'results'>('console');

  const handleAnalysisComplete = (subphaseId: string, result: AnalysisResponse) => {
    setSubphaseResults(prev => ({
      ...prev,
      [subphaseId]: {
        ...prev[subphaseId],
        analysis: result,
        logs: result.logs
      }
    }));
    setActiveSubphase(subphaseId);
    setActiveTab('results');
  };

  const handleValidationComplete = (subphaseId: string, result: ValidationResponse) => {
    setSubphaseResults(prev => ({
      ...prev,
      [subphaseId]: {
        ...prev[subphaseId],
        validation: result,
        logs: result.logs
      }
    }));
    setActiveSubphase(subphaseId);
    setActiveTab('results');
  };

  const currentResults = subphaseResults[activeSubphase] || { logs: [] };

  return (
    <div className="flex h-screen bg-gray-100">
      <ValidationSidebar
        onAnalysisComplete={handleAnalysisComplete}
        onValidationComplete={handleValidationComplete}
      />

      <main className="flex-1 overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Lending Analysis Dashboard
            </h1>
            <p className="text-gray-600">
              Run and validate analysis subphases • Phases 1-4: EDA → Features → Lenders → Revenue
            </p>
          </div>

          {/* Subphase Navigation */}
          <div className="mb-4">
            <div className="flex flex-wrap gap-2 mb-4">
              {/* Phase 1 */}
              {['phase1_1', 'phase1_2', 'phase1_3'].map(subphaseId => {
                const hasResults = subphaseResults[subphaseId];
                const label = subphaseId === 'phase1_1' ? '1.1 Univariate' :
                             subphaseId === 'phase1_2' ? '1.2 Bivariate' : '1.3 Missing Values';

                return (
                  <button
                    key={subphaseId}
                    onClick={() => setActiveSubphase(subphaseId)}
                    disabled={!hasResults}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      activeSubphase === subphaseId
                        ? 'bg-blue-500 text-white'
                        : hasResults
                        ? 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-200'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    {label} {hasResults && '✓'}
                  </button>
                );
              })}

              {/* Phase 2 */}
              {['phase2_1', 'phase2_2', 'phase2_3'].map(subphaseId => {
                const hasResults = subphaseResults[subphaseId];
                const label = subphaseId === 'phase2_1' ? '2.1 Statistical' :
                             subphaseId === 'phase2_2' ? '2.2 ML Importance' : '2.3 Feature Validation';

                return (
                  <button
                    key={subphaseId}
                    onClick={() => setActiveSubphase(subphaseId)}
                    disabled={!hasResults}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      activeSubphase === subphaseId
                        ? 'bg-green-500 text-white'
                        : hasResults
                        ? 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-200'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    {label} {hasResults && '✓'}
                  </button>
                );
              })}

              {/* Phase 3 */}
              {['phase3_1', 'phase3_2', 'phase3_3'].map(subphaseId => {
                const hasResults = subphaseResults[subphaseId];
                const label = subphaseId === 'phase3_1' ? '3.1 Lender Profiling' :
                             subphaseId === 'phase3_2' ? '3.2 Lender Models' : '3.3 Specialization';

                return (
                  <button
                    key={subphaseId}
                    onClick={() => setActiveSubphase(subphaseId)}
                    disabled={!hasResults}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      activeSubphase === subphaseId
                        ? 'bg-purple-500 text-white'
                        : hasResults
                        ? 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-200'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    {label} {hasResults && '✓'}
                  </button>
                );
              })}

              {/* Phase 4 */}
              {[
                { id: 'phase4_1', label: '4.1 Baseline Revenue' },
                { id: 'phase4_2', label: '4.2 Optimal Matching' },
                { id: 'phase4_3', label: '4.3 Incremental Revenue' }
              ].map(({ id: subphaseId, label }) => {
                const hasResults = subphaseResults[subphaseId];

                return (
                  <button
                    key={subphaseId}
                    onClick={() => setActiveSubphase(subphaseId)}
                    disabled={!hasResults}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      activeSubphase === subphaseId
                        ? 'bg-orange-500 text-white'
                        : hasResults
                        ? 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-200'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    {label} {hasResults && '✓'}
                  </button>
                );
              })}
            </div>

            {/* Tab Navigation */}
            <div className="border-b border-gray-200">
              <div className="flex gap-4">
                <button
                  onClick={() => setActiveTab('console')}
                  className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                    activeTab === 'console'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  💻 Console Output
                </button>
                <button
                  onClick={() => setActiveTab('results')}
                  className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                    activeTab === 'results'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  📊 Results
                </button>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="space-y-6">
            {activeTab === 'console' && <ConsoleOutput logs={currentResults.logs} />}

            {activeTab === 'results' && (
              <ValidationResults
                analysisResult={currentResults.analysis}
                validationResult={currentResults.validation}
              />
            )}
          </div>

          {/* Quick Guide */}
          <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">🚀 Quick Start Guide</h3>
            <ol className="space-y-1 text-sm text-blue-800 list-decimal list-inside">
              <li>Click <strong>"▶ Run"</strong> on any subphase (1.1, 1.2, or 1.3) to execute analysis</li>
              <li>View real-time console output in the <strong>Console Output</strong> tab</li>
              <li>Switch to <strong>Results</strong> tab to see key insights and generated files</li>
              <li>Click <strong>"✓ Validate"</strong> to run validation tests (verifies all outputs exist)</li>
              <li>Check the status indicator: ✅ = Passed, ❌ = Failed, 🔄 = Running, ⏳ = Pending</li>
            </ol>
            <div className="mt-3 pt-3 border-t border-blue-200">
              <p className="text-xs text-blue-700">
                <strong>Phase 1 Complete:</strong> Run all 3 subphases (1.1 → 1.2 → 1.3) and validate each to complete Phase 1: EDA
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
