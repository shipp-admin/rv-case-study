'use client';

import { ValidationResponse, AnalysisResponse } from '@/lib/types';
import FigureGallery from './FigureGallery';
import TablePreview from './TablePreview';

interface ValidationResultsProps {
  validationResult?: ValidationResponse;
  analysisResult?: AnalysisResponse;
}

export default function ValidationResults({
  validationResult,
  analysisResult,
}: ValidationResultsProps) {
  // Detect phase path from subphase name
  const getPhasePath = (subphase: string): string => {
    if (subphase.includes('Phase 1') || subphase.includes('1.')) return 'phase1_eda';
    if (subphase.includes('Phase 2') || subphase.includes('2.')) return 'phase2_feature_importance';
    if (subphase.includes('Phase 3') || subphase.includes('3.')) return 'phase3_lender_analysis';
    if (subphase.includes('Phase 4') || subphase.includes('4.')) return 'phase4_revenue_optimization';
    return 'phase1_eda'; // default
  };

  const phasePath = analysisResult?.result?.subphase
    ? getPhasePath(analysisResult.result.subphase)
    : 'phase1_eda';

  if (!validationResult && !analysisResult) {
    return (
      <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
        <p className="text-gray-500 italic text-center">
          Run an analysis or validation to see results here.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Analysis Results */}
      {analysisResult?.result && (
        <div className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
          <div className="bg-blue-50 px-4 py-3 border-b border-blue-100">
            <h3 className="text-lg font-semibold text-blue-900">
              {analysisResult.result.subphase}
            </h3>
            <p className="text-sm text-blue-700 mt-1">
              Analysis completed in {analysisResult.result.execution_time}s
            </p>
          </div>

          <div className="p-4">
            {/* Summary */}
            {analysisResult.result.summary && (
              <div className="mb-4">
                <h4 className="font-semibold text-gray-700 mb-2">Summary</h4>
                <div className="bg-gray-50 rounded p-3 space-y-1 text-sm">
                  {analysisResult.result.summary.overall_approval_rate !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Approval Rate:</span>
                      <span className="font-medium">
                        {(analysisResult.result.summary.overall_approval_rate * 100).toFixed(2)}%
                      </span>
                    </div>
                  )}
                  {analysisResult.result.summary.variables_analyzed && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Categorical Variables:</span>
                        <span className="font-medium">
                          {analysisResult.result.summary.variables_analyzed.categorical}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Numerical Variables:</span>
                        <span className="font-medium">
                          {analysisResult.result.summary.variables_analyzed.numerical}
                        </span>
                      </div>
                    </>
                  )}
                  {/* Phase 4.1: Revenue metrics */}
                  {analysisResult.result.summary.overall_rpa !== undefined && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Overall RPA:</span>
                        <span className="font-medium">
                          ${analysisResult.result.summary.overall_rpa.toFixed(2)}
                        </span>
                      </div>
                      {analysisResult.result.summary.total_revenue && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Total Revenue:</span>
                          <span className="font-medium">
                            ${analysisResult.result.summary.total_revenue.toLocaleString()}
                          </span>
                        </div>
                      )}
                      {analysisResult.result.summary.lender_rpa_range && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Lender RPA Range:</span>
                          <span className="font-medium">
                            ${analysisResult.result.summary.lender_rpa_range.min} - ${analysisResult.result.summary.lender_rpa_range.max}
                          </span>
                        </div>
                      )}
                    </>
                  )}

                  {/* Phase 4.2: Matching algorithm metrics */}
                  {analysisResult.result.summary.mean_optimal_ev !== undefined && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Mean Optimal EV:</span>
                        <span className="font-medium">
                          ${analysisResult.result.summary.mean_optimal_ev.toFixed(2)}
                        </span>
                      </div>
                      {analysisResult.result.summary.mean_assignment_confidence !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Assignment Confidence:</span>
                          <span className="font-medium">
                            {(analysisResult.result.summary.mean_assignment_confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {analysisResult.result.summary.pct_should_switch !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Should Switch:</span>
                          <span className="font-medium">
                            {analysisResult.result.summary.pct_should_switch.toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {analysisResult.result.summary.latency_per_customer_ms !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Latency:</span>
                          <span className="font-medium">
                            {analysisResult.result.summary.latency_per_customer_ms.toFixed(2)}ms
                          </span>
                        </div>
                      )}
                    </>
                  )}

                  {/* Phase 4.3: Incremental Revenue metrics */}
                  {analysisResult.result.summary.incremental_revenue !== undefined && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Baseline Revenue:</span>
                        <span className="font-medium">
                          ${analysisResult.result.summary.baseline_total_revenue?.toLocaleString() || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Optimal Revenue:</span>
                        <span className="font-medium">
                          ${analysisResult.result.summary.optimal_total_revenue?.toLocaleString() || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Incremental Revenue:</span>
                        <span className="font-medium text-green-600">
                          ${analysisResult.result.summary.incremental_revenue.toLocaleString()}
                        </span>
                      </div>
                      {analysisResult.result.summary.lift_percentage !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Revenue Lift:</span>
                          <span className="font-medium text-green-600">
                            +{analysisResult.result.summary.lift_percentage.toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {analysisResult.result.summary.ci_lower !== undefined && analysisResult.result.summary.ci_upper !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">95% CI:</span>
                          <span className="font-medium text-xs">
                            [${analysisResult.result.summary.ci_lower.toLocaleString()}, ${analysisResult.result.summary.ci_upper.toLocaleString()}]
                          </span>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Insights */}
            {analysisResult.result.insights && analysisResult.result.insights.length > 0 && (
              <div className="mb-4">
                <h4 className="font-semibold text-gray-700 mb-2">💡 Key Insights</h4>
                <ul className="space-y-2">
                  {analysisResult.result.insights.map((insight, index) => (
                    <li
                      key={index}
                      className="flex items-start gap-2 text-sm text-gray-700 bg-yellow-50 p-2 rounded"
                    >
                      <span className="text-yellow-600 mt-0.5">•</span>
                      <span>{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Outputs */}
            {analysisResult.result.outputs && (
              <div className="space-y-4">
                {/* Figures Gallery */}
                {analysisResult.result.outputs.figures && analysisResult.result.outputs.figures.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-700 mb-3">
                      📈 Figures ({analysisResult.result.outputs.figures.length})
                    </h4>
                    <FigureGallery figures={analysisResult.result.outputs.figures} phasePath={phasePath} />
                  </div>
                )}

                {/* Tables Preview */}
                {analysisResult.result.outputs.tables && analysisResult.result.outputs.tables.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-700 mb-2">
                      📊 Tables ({analysisResult.result.outputs.tables.length})
                    </h4>
                    <div className="bg-gray-50 rounded p-3">
                      <TablePreview tables={analysisResult.result.outputs.tables} phasePath={phasePath} />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Validation Results */}
      {validationResult && (
        <div className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
          <div
            className={`px-4 py-3 border-b ${
              validationResult.success
                ? 'bg-green-50 border-green-100'
                : 'bg-red-50 border-red-100'
            }`}
          >
            <div className="flex items-center justify-between">
              <h3
                className={`text-lg font-semibold ${
                  validationResult.success ? 'text-green-900' : 'text-red-900'
                }`}
              >
                {validationResult.success ? '✅ VALIDATION PASSED' : '❌ VALIDATION FAILED'}
              </h3>
              <div className="text-right">
                <div
                  className={`text-2xl font-bold ${
                    validationResult.success ? 'text-green-700' : 'text-red-700'
                  }`}
                >
                  {(validationResult.pass_rate * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-gray-600">
                  {validationResult.checks_passed} / {validationResult.total_checks} checks
                </div>
              </div>
            </div>
          </div>

          <div className="p-4">
            {/* Validation Details */}
            {validationResult.details && validationResult.details.length > 0 && (
              <div className="space-y-3 mb-4">
                {validationResult.details.map((detail, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg overflow-hidden">
                    <div className="bg-gray-50 px-3 py-2 border-b border-gray-200">
                      <h4 className="font-semibold text-gray-700 text-sm">
                        {detail.category} ({detail.checks.filter((c) => c.passed).length}/
                        {detail.checks.length})
                      </h4>
                    </div>
                    <div className="p-3 space-y-1">
                      {detail.checks.map((check, checkIndex) => (
                        <div
                          key={checkIndex}
                          className="flex items-center gap-2 text-sm"
                        >
                          <span className={check.passed ? 'text-green-600' : 'text-red-600'}>
                            {check.passed ? '✓' : '✗'}
                          </span>
                          <span className="text-gray-700">{check.name}</span>
                          <span className="text-gray-500 text-xs ml-auto">{check.message}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Validation Insights */}
            {validationResult.insights && validationResult.insights.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-700 mb-2">💡 Validation Insights</h4>
                <ul className="space-y-2">
                  {validationResult.insights.map((insight, index) => (
                    <li
                      key={index}
                      className="flex items-start gap-2 text-sm text-gray-700 bg-blue-50 p-2 rounded"
                    >
                      <span className="text-blue-600 mt-0.5">•</span>
                      <span>{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
