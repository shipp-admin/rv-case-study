'use client';

import { useState } from 'react';
import { Subphase, SubphaseStatus, AnalysisResponse, ValidationResponse } from '@/lib/types';

interface ValidationSidebarProps {
  onAnalysisComplete?: (subphaseId: string, result: AnalysisResponse) => void;
  onValidationComplete?: (subphaseId: string, result: ValidationResponse) => void;
}

export default function ValidationSidebar({
  onAnalysisComplete,
  onValidationComplete,
}: ValidationSidebarProps) {
  const [subphases, setSubphases] = useState<Subphase[]>([
    // Phase 1: EDA
    {
      id: 'phase1_1',
      name: '1.1 Univariate',
      phase: 1,
      status: 'pending',
      script: 'src/phase1_eda/univariate.py',
      test: 'tests/test_phase1_univariate.py',
    },
    {
      id: 'phase1_2',
      name: '1.2 Bivariate',
      phase: 1,
      status: 'pending',
      script: 'src/phase1_eda/bivariate.py',
      test: 'tests/test_phase1_bivariate.py',
    },
    {
      id: 'phase1_3',
      name: '1.3 Missing Values',
      phase: 1,
      status: 'pending',
      script: 'src/phase1_eda/missing_values.py',
      test: 'tests/test_phase1_missing_values.py',
    },
    // Phase 2: Feature Importance
    {
      id: 'phase2_1',
      name: '2.1 Statistical',
      phase: 2,
      status: 'pending',
      script: 'src/phase2_feature_importance/statistical_importance.py',
      test: 'tests/test_phase2_statistical_importance.py',
    },
    {
      id: 'phase2_2',
      name: '2.2 ML Importance',
      phase: 2,
      status: 'pending',
      script: 'src/phase2_feature_importance/ml_importance.py',
      test: 'tests/test_phase2_ml_importance.py',
    },
    {
      id: 'phase2_3',
      name: '2.3 Feature Validation',
      phase: 2,
      status: 'pending',
      script: 'src/phase2_feature_importance/feature_validation.py',
      test: 'tests/test_phase2_feature_validation.py',
    },
    // Phase 3: Lender Analysis
    {
      id: 'phase3_1',
      name: '3.1 Lender Profiling',
      phase: 3,
      status: 'pending',
      script: 'src/phase3_lender_analysis/lender_profiling.py',
      test: 'tests/test_phase3_lender_profiling.py',
    },
    {
      id: 'phase3_2',
      name: '3.2 Lender Models',
      phase: 3,
      status: 'pending',
      script: 'src/phase3_lender_analysis/lender_models.py',
      test: 'tests/test_phase3_lender_models.py',
    },
    {
      id: 'phase3_3',
      name: '3.3 Specialization',
      phase: 3,
      status: 'pending',
      script: 'src/phase3_lender_analysis/lender_specialization.py',
      test: 'tests/test_phase3_specialization.py',
    },
    // Phase 4: Revenue Optimization
    {
      id: 'phase4_1',
      name: '4.1 Baseline Revenue',
      phase: 4,
      status: 'pending',
      script: 'src/phase4_revenue_optimization/baseline_revenue.py',
      test: 'tests/test_phase4_baseline_revenue.py',
    },
    {
      id: 'phase4_2',
      name: '4.2 Optimal Matching',
      phase: 4,
      status: 'pending',
      script: 'src/phase4_revenue_optimization/matching_algorithm.py',
      test: 'tests/test_phase4_matching_algorithm.py',
    },
    {
      id: 'phase4_3',
      name: '4.3 Incremental Revenue',
      phase: 4,
      status: 'pending',
      script: 'src/phase4_revenue_optimization/incremental_revenue.py',
      test: 'tests/test_phase4_incremental_revenue.py',
    },
  ]);

  const [activeSubphase, setActiveSubphase] = useState<string | null>(null);

  const getStatusIcon = (status: SubphaseStatus): string => {
    switch (status) {
      case 'passed':
        return '✅';
      case 'failed':
        return '❌';
      case 'running':
        return '🔄';
      case 'completed':
        return '✓';
      case 'warning':
        return '⚠️';
      default:
        return '⏳';
    }
  };

  const getStatusColor = (status: SubphaseStatus): string => {
    switch (status) {
      case 'passed':
        return 'text-green-600';
      case 'failed':
        return 'text-red-600';
      case 'running':
        return 'text-blue-600';
      case 'warning':
        return 'text-yellow-600';
      default:
        return 'text-gray-400';
    }
  };

  const runAnalysis = async (subphase: Subphase) => {
    setActiveSubphase(subphase.id);
    setSubphases((prev) =>
      prev.map((s) =>
        s.id === subphase.id ? { ...s, status: 'running' as SubphaseStatus } : s
      )
    );

    try {
      const res = await fetch('/api/analysis/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script: subphase.script }),
      });

      const result: AnalysisResponse = await res.json();

      setSubphases((prev) =>
        prev.map((s) =>
          s.id === subphase.id
            ? {
                ...s,
                status: result.success ? 'completed' : 'failed',
                analysisResult: result.result || undefined,
              }
            : s
        )
      );

      onAnalysisComplete?.(subphase.id, result);
    } catch (error) {
      console.error('Analysis error:', error);
      setSubphases((prev) =>
        prev.map((s) =>
          s.id === subphase.id ? { ...s, status: 'failed' as SubphaseStatus } : s
        )
      );
    }
  };

  const runValidation = async (subphase: Subphase) => {
    setActiveSubphase(subphase.id);
    setSubphases((prev) =>
      prev.map((s) =>
        s.id === subphase.id ? { ...s, status: 'running' as SubphaseStatus } : s
      )
    );

    try {
      const res = await fetch('/api/validation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test: subphase.test }),
      });

      const result: ValidationResponse = await res.json();

      setSubphases((prev) =>
        prev.map((s) =>
          s.id === subphase.id
            ? {
                ...s,
                status: result.success ? 'passed' : 'failed',
                validationResult: result,
              }
            : s
        )
      );

      onValidationComplete?.(subphase.id, result);
    } catch (error) {
      console.error('Validation error:', error);
      setSubphases((prev) =>
        prev.map((s) =>
          s.id === subphase.id ? { ...s, status: 'failed' as SubphaseStatus } : s
        )
      );
    }
  };

  const phaseGroups = [
    { phase: 1, title: 'Phase 1: EDA', description: 'Exploratory Data Analysis' },
    { phase: 2, title: 'Phase 2: Feature Importance', description: 'Statistical & ML Analysis' },
    { phase: 3, title: 'Phase 3: Lender Analysis', description: 'Lender-Specific Profiling' },
    { phase: 4, title: 'Phase 4: Revenue Optimization', description: 'Baseline & Matching' },
  ];

  return (
    <aside className="w-64 bg-gray-50 border-r border-gray-200 p-4 overflow-y-auto">
      <h2 className="text-lg font-bold mb-4 text-gray-800">Analysis Pipeline</h2>

      <div className="space-y-6">
        {phaseGroups.map((group) => {
          const phaseSubphases = subphases.filter((s) => s.phase === group.phase);

          return (
            <div key={group.phase} className={group.disabled ? 'opacity-50' : ''}>
              <div className="mb-3">
                <h3 className="font-semibold text-sm text-gray-700 uppercase tracking-wide">
                  {group.title}
                </h3>
                {group.disabled && (
                  <p className="text-xs text-gray-500 mt-1">{group.description}</p>
                )}
              </div>

              {!group.disabled && phaseSubphases.length > 0 && (
                <div className="space-y-3">
                  {phaseSubphases.map((subphase) => (
                <div
                  key={subphase.id}
                  className={`p-3 bg-white rounded-lg shadow-sm border ${
                    activeSubphase === subphase.id
                      ? 'border-blue-400 ring-2 ring-blue-100'
                      : 'border-gray-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">
                      {subphase.name}
                    </span>
                    <span className={`text-lg ${getStatusColor(subphase.status)}`}>
                      {getStatusIcon(subphase.status)}
                    </span>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => runAnalysis(subphase)}
                      disabled={subphase.status === 'running'}
                      className="flex-1 text-xs bg-blue-500 hover:bg-blue-600 text-white px-2 py-1.5 rounded disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                    >
                      {subphase.status === 'running' ? '⏳ Running...' : '▶ Run'}
                    </button>
                    <button
                      onClick={() => runValidation(subphase)}
                      disabled={subphase.status === 'running'}
                      className="flex-1 text-xs bg-green-500 hover:bg-green-600 text-white px-2 py-1.5 rounded disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                    >
                      {subphase.status === 'running' ? '⏳ Running...' : '✓ Validate'}
                    </button>
                  </div>

                  {subphase.analysisResult && (
                    <div className="mt-2 text-xs text-gray-600">
                      <div className="flex justify-between">
                        <span>Execution:</span>
                        <span className="font-medium">
                          {subphase.analysisResult.execution_time}s
                        </span>
                      </div>
                    </div>
                  )}

                  {subphase.validationResult && (
                    <div className="mt-2 text-xs text-gray-600">
                      <div className="flex justify-between">
                        <span>Pass Rate:</span>
                        <span
                          className={`font-medium ${
                            subphase.validationResult.pass_rate >= 0.8
                              ? 'text-green-600'
                              : 'text-red-600'
                          }`}
                        >
                          {(subphase.validationResult.pass_rate * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Checks:</span>
                        <span className="font-medium">
                          {subphase.validationResult.checks_passed}/
                          {subphase.validationResult.total_checks}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
