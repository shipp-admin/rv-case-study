'use client';

import { useEffect, useRef } from 'react';

interface ConsoleOutputProps {
  logs: string[];
  title?: string;
}

export default function ConsoleOutput({ logs, title = 'Console Output' }: ConsoleOutputProps) {
  const outputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom when new logs arrive
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
      <div className="bg-gray-800 px-4 py-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-100">{title}</h3>
        <span className="text-xs text-gray-400">{logs.length} lines</span>
      </div>

      <div
        ref={outputRef}
        className="bg-gray-900 p-4 font-mono text-xs text-green-400 h-96 overflow-y-auto"
      >
        {logs.length === 0 ? (
          <div className="text-gray-500 italic">
            No output yet. Run an analysis to see results here.
          </div>
        ) : (
          logs.map((log, index) => (
            <div key={index} className="whitespace-pre-wrap">
              {log}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
