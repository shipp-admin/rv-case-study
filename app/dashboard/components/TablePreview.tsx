'use client';

import { useState, useEffect } from 'react';

interface TablePreviewProps {
  tables: string[];
  basePath?: string;
  phasePath?: string;
}

interface TableData {
  headers: string[];
  rows: string[][];
}

export default function TablePreview({ tables, basePath = 'tables', phasePath }: TablePreviewProps) {
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [tableData, setTableData] = useState<TableData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (selectedTable) {
      loadTableData(selectedTable);
    }
  }, [selectedTable]);

  const loadTableData = async (tableName: string) => {
    setLoading(true);
    setError(null);
    try {
      const phase = phasePath || 'phase1_eda';
      const response = await fetch(`/${phase}/${basePath}/${tableName}`);
      if (!response.ok) throw new Error('Failed to load table');

      const text = await response.text();
      const lines = text.trim().split('\n');
      const headers = lines[0].split(',').map(h => h.trim());
      const rows = lines.slice(1).map(line =>
        line.split(',').map(cell => cell.trim())
      );

      setTableData({ headers, rows });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getTableTitle = (filename: string) => {
    return filename
      .replace('.csv', '')
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (tables.length === 0) {
    return (
      <div className="text-gray-500 italic text-sm">
        No tables generated
      </div>
    );
  }

  return (
    <>
      <div className="space-y-2">
        {tables.map((table, index) => (
          <div key={index} className="flex items-center gap-2">
            <span className="text-green-600">✓</span>
            <button
              onClick={() => setSelectedTable(table)}
              className="text-blue-600 hover:underline text-sm"
            >
              {getTableTitle(table)}
            </button>
            <a
              href={`/${phasePath || 'phase1_eda'}/${basePath}/${table}`}
              download
              className="ml-auto text-xs text-gray-500 hover:text-gray-700"
            >
              ⬇ Download
            </a>
          </div>
        ))}
      </div>

      {/* Modal for table preview */}
      {selectedTable && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedTable(null)}
        >
          <div
            className="relative max-w-6xl max-h-[90vh] bg-white rounded-lg overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="bg-gray-50 px-4 py-3 border-b border-gray-200 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">
                {getTableTitle(selectedTable)}
              </h3>
              <button
                onClick={() => setSelectedTable(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-4 overflow-auto max-h-[calc(90vh-8rem)]">
              {loading && (
                <div className="text-center py-8 text-gray-500">
                  Loading table data...
                </div>
              )}

              {error && (
                <div className="text-center py-8 text-red-600">
                  Error: {error}
                </div>
              )}

              {tableData && !loading && !error && (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 border border-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        {tableData.headers.map((header, idx) => (
                          <th
                            key={idx}
                            className="px-4 py-2 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r border-gray-200 last:border-r-0"
                          >
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {tableData.rows.slice(0, 50).map((row, rowIdx) => (
                        <tr key={rowIdx} className="hover:bg-gray-50">
                          {row.map((cell, cellIdx) => (
                            <td
                              key={cellIdx}
                              className="px-4 py-2 text-sm text-gray-900 border-r border-gray-200 last:border-r-0"
                            >
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {tableData.rows.length > 50 && (
                    <div className="text-center py-2 text-sm text-gray-500 bg-gray-50 border-t">
                      Showing first 50 of {tableData.rows.length} rows
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="bg-gray-50 px-4 py-3 border-t border-gray-200 flex justify-end">
              <a
                href={`/${phasePath || 'phase1_eda'}/${basePath}/${selectedTable}`}
                download
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
              >
                Download CSV
              </a>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
